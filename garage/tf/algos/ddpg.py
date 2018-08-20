"""
This module implements a DDPG model.

DDPG, also known as Deep Deterministic Policy Gradient, uses actor-critic
method to optimize the policy and reward prediction. It uses a supervised
method to update the critic network and policy gradient to update the actor
network. And there are exploration strategy, replay buffer and target networks
involved to stabilize the training process.
"""
from collections import deque
from copy import copy

import numpy as np
import pyprind
import tensorflow as tf
import tensorflow.contrib as tc

from garage.algos.base import RLAlgorithm
from garage.envs.util import configure_dims, dims_to_shapes
from garage.misc import logger
from garage.misc.overrides import overrides
from garage.replay_buffer import HerReplayBuffer
from garage.replay_buffer import ReplayBuffer
from garage.replay_buffer.her_replay_buffer import make_her_sample
from garage.tf.misc import tensor_utils
from garage.tf.plotter import Plotter


class DDPG(RLAlgorithm):
    """
    A DDPG model based on https://arxiv.org/pdf/1509.02971.pdf.

    Example:
        $ python garage/examples/tf/ddpg_pendulum.py
    """

    def __init__(self,
                 env,
                 actor,
                 critic,
                 n_epochs=500,
                 n_epoch_cycles=20,
                 n_rollout_steps=100,
                 n_train_steps=50,
                 reward_scale=1.,
                 batch_size=64,
                 target_update_tau=0.01,
                 discount=0.99,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 actor_weight_decay=0,
                 critic_weight_decay=0,
                 replay_buffer_size=int(1e6),
                 min_buffer_size=10000,
                 exploration_strategy=None,
                 plot=False,
                 pause_for_plot=False,
                 actor_optimizer=None,
                 critic_optimizer=None,
                 use_her=False,
                 clip_obs=np.inf,
                 clip_pos_returns=True,
                 clip_return=None,
                 replay_k=4,
                 max_action=None,
                 name=None):
        """
        Construct class.

        Args:
            env(): Environment.
            actor(garage.tf.policies.ContinuousMLPPolicy): Policy network.
            critic(garage.tf.q_functions.ContinuousMLPQFunction):
        Q Value network.
            n_epochs(int, optional): Number of epochs.
            n_epoch_cycles(int, optional): Number of epoch cycles.
            n_rollout_steps(int, optional): Number of rollout steps.
        aka the time horizon of rollout.
            n_train_steps(int, optional): Number of train steps.
            reward_scale(float): The scaling factor applied to the rewards when
        training.
            batch_size(int): Number of samples for each minibatch.
            target_update_tau(float): Interpolation parameter for doing the
        soft target update.
            discount(float): Discount factor for the cumulative return.
            actor_lr(float): Learning rate for training policy network.
            critic_lr(float): Learning rate for training q value network.
            actor_weight_decay(float): L2 weight decay factor for parameters of
        the policy network.
            critic_weight_decay(float): L2 weight decay factor for parameters
        of the q value network.
            replay_buffer_size(int): Size of the replay buffer.
            min_buffer_size(int): Minimum size of the replay buffer to start
        training.
            exploration_strategy(): Exploration strategy to randomize the
        action.
            plot(boolean): Whether to visualize the policy performance after
        each eval_interval.
            pause_for_plot(boolean): Whether or not pause before continuing
        when plotting.
            actor_optimizer(): Optimizer for training policy network.
            critic_optimizer(): Optimizer for training q function network.
            use_her(boolean): Whether or not use HER for replay buffer.
            clip_obs(float): Clip observation to be in [-clip_obs, clip_obs].
            clip_pos_returns(boolean): Whether or not clip positive returns.
            clip_return(float): Clip return to be in [-clip_return,
        clip_return].
            replay_k(int): The ratio between HER replays and regular replays.
        Only used when use_her is True.
            max_action(float): Maximum action magnitude.
            name(str): Name of the algorithm shown in computation graph.
        """
        self.env = env

        self.input_dims = configure_dims(env)
        action_bound = env.action_space.high
        self.max_action = action_bound if max_action is None else max_action

        self.actor = actor
        self.critic = critic
        self.n_epochs = n_epochs
        self.n_epoch_cycles = n_epoch_cycles
        self.n_rollout_steps = n_rollout_steps
        self.n_train_steps = n_train_steps
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.tau = target_update_tau
        self.discount = discount
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_weight_decay = actor_weight_decay
        self.critic_weight_decay = critic_weight_decay
        self.replay_buffer_size = replay_buffer_size
        self.min_buffer_size = min_buffer_size
        self.es = exploration_strategy
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.name = name
        self.use_her = use_her
        self.evaluate = False
        self.replay_k = replay_k
        self.clip_return = (
            1. / (1. - self.discount)) if clip_return is None else clip_return
        self.clip_obs = clip_obs
        self.clip_pos_returns = clip_pos_returns
        self.success_history = deque(maxlen=100)
        self._initialize()

    @overrides
    def train(self, sess=None):
        """
        Training process of DDPG algorithm.

        Args:
            sess: A TensorFlow session for executing ops.
        """
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        # Start plotter
        if self.plot:
            self.plotter = Plotter(self.env, self.actor, sess)
            self.plotter.start()

        sess.run(tf.global_variables_initializer())
        self.f_init_target()

        observation = self.env.reset()
        if self.es:
            self.es.reset()

        episode_reward = 0.
        episode_step = 0
        episode_rewards = []
        episode_steps = []
        episode_actor_losses = []
        episode_critic_losses = []
        episodes = 0
        epoch_ys = []
        epoch_qs = []

        for epoch in range(self.n_epochs):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Training started")
            self.success_history.clear()
            for epoch_cycle in pyprind.prog_bar(range(self.n_epoch_cycles)):
                if self.use_her:
                    successes = []
                    for rollout in range(self.n_rollout_steps):
                        o = np.clip(observation["observation"], -self.clip_obs,
                                    self.clip_obs)
                        g = np.clip(observation["desired_goal"],
                                    -self.clip_obs, self.clip_obs)
                        obs_goal = np.concatenate((o, g), axis=-1)
                        action = self.es.get_action(rollout, obs_goal,
                                                    self.actor)

                        next_observation, reward, terminal, info = self.env.step(  # noqa: E501
                            action)
                        if 'is_success' in info:
                            successes.append([info["is_success"]])
                        episode_reward += reward
                        episode_step += 1

                        info_dict = {
                            "info_{}".format(key): info[key].reshape(1)
                            for key in info.keys()
                        }
                        self.replay_buffer.add_transition(
                            observation=observation['observation'],
                            action=action,
                            goal=observation['desired_goal'],
                            achieved_goal=observation['achieved_goal'],
                            **info_dict,
                        )

                        observation = next_observation

                        if rollout == self.n_rollout_steps - 1:
                            self.replay_buffer.add_transition(
                                observation=observation['observation'],
                                achieved_goal=observation['achieved_goal'])

                            episode_rewards.append(episode_reward)
                            episode_steps.append(episode_step)
                            episode_reward = 0.
                            episode_step = 0
                            episodes += 1

                            observation = self.env.reset()
                            if self.es:
                                self.es.reset()

                    successful = np.array(successes)[-1, :]
                    success_rate = np.mean(successful)
                    self.success_history.append(success_rate)

                    for train_itr in range(self.n_train_steps):
                        self.evaluate = True
                        critic_loss, y, q, action_loss = self._learn()

                        episode_actor_losses.append(action_loss)
                        episode_critic_losses.append(critic_loss)
                        epoch_ys.append(y)
                        epoch_qs.append(q)

                    self.f_update_target()
                else:
                    for rollout in range(self.n_rollout_steps):
                        action = self.es.get_action(rollout, observation,
                                                    self.actor)
                        assert action.shape == self.env.action_space.shape

                        next_observation, reward, terminal, info = self.env.step(  # noqa: E501
                            action)
                        episode_reward += reward
                        episode_step += 1

                        self.replay_buffer.add_transition(
                            observation=observation,
                            action=action,
                            reward=reward * self.reward_scale,
                            terminal=terminal,
                            next_observation=next_observation,
                        )

                        observation = next_observation

                        if terminal or rollout == self.n_rollout_steps - 1:
                            episode_rewards.append(episode_reward)
                            episode_steps.append(episode_step)
                            episode_reward = 0.
                            episode_step = 0
                            episodes += 1

                            observation = self.env.reset()
                            if self.es:
                                self.es.reset()

                    for train_itr in range(self.n_train_steps):
                        if self.replay_buffer.size >= self.min_buffer_size:
                            self.evaluate = True
                            critic_loss, y, q, action_loss = self._learn()

                            episode_actor_losses.append(action_loss)
                            episode_critic_losses.append(critic_loss)
                            epoch_ys.append(y)
                            epoch_qs.append(q)

            logger.log("Training finished")
            logger.log("Saving snapshot")
            itr = epoch * self.n_epoch_cycles + epoch_cycle
            params = self.get_itr_snapshot(itr)
            logger.save_itr_params(itr, params)
            logger.log("Saved")
            if self.evaluate:
                logger.record_tabular('Epoch', epoch)
                logger.record_tabular('Episodes', episodes)
                logger.record_tabular('AverageReturn',
                                      np.mean(episode_rewards))
                logger.record_tabular('StdReturn', np.std(episode_rewards))
                logger.record_tabular('Policy/AveragePolicyLoss',
                                      np.mean(episode_actor_losses))
                logger.record_tabular('QFunction/AverageQFunctionLoss',
                                      np.mean(episode_critic_losses))
                logger.record_tabular('QFunction/AverageQ', np.mean(epoch_qs))
                logger.record_tabular('QFunction/MaxQ', np.max(epoch_qs))
                logger.record_tabular('QFunction/AverageAbsQ',
                                      np.mean(np.abs(epoch_qs)))
                logger.record_tabular('QFunction/AverageY', np.mean(epoch_ys))
                logger.record_tabular('QFunction/MaxY', np.max(epoch_ys))
                logger.record_tabular('QFunction/AverageAbsY',
                                      np.mean(np.abs(epoch_ys)))
                if self.use_her:
                    logger.record_tabular('AverageSuccessRate',
                                          np.mean(self.success_history))

                # Uncomment the following if you want to calculate the average
                # in each epoch, better uncomment when self.use_her is True
                # episode_rewards = []
                # episode_actor_losses = []
                # episode_critic_losses = []
                # epoch_ys = []
                # epoch_qs = []

            logger.dump_tabular(with_prefix=False)
            logger.pop_prefix()
            if self.plot:
                self.plotter.update_plot(self.actor, self.n_rollout_steps)
                if self.pause_for_plot:
                    input("Plotting evaluation run: Press Enter to "
                          "continue...")

        if self.plot:
            self.plotter.shutdown()
        if created_session:
            sess.close()

    def _initialize(self):
        with tf.name_scope(self.name, "DDPG"):
            with tf.name_scope("setup_networks"):
                """Set up the actor, critic and target network."""
                # Set up the actor and critic network
                self.actor._build_net(trainable=True)
                self.critic._build_net(trainable=True)

                # Create target actor and critic network
                target_actor = copy(self.actor)
                target_critic = copy(self.critic)

                # Set up the target network
                target_actor.name = "TargetActor"
                target_actor._build_net(trainable=False)
                target_critic.name = "TargetCritic"
                target_critic._build_net(trainable=False)

            input_shapes = dims_to_shapes(self.input_dims)

            # Initialize replay buffer
            if self.use_her:
                buffer_shapes = {
                    key: (self.n_rollout_steps + 1
                          if key == "observation" or key == "achieved_goal"
                          else self.n_rollout_steps, *input_shapes[key])
                    for key, val in input_shapes.items()
                }

                replay_buffer = HerReplayBuffer(
                    buffer_shapes=buffer_shapes,
                    size_in_transitions=self.replay_buffer_size,
                    time_horizon=self.n_rollout_steps,
                    sample_transitions=make_her_sample(
                        self.replay_k, self.env.compute_reward))
            else:
                replay_buffer = ReplayBuffer(
                    buffer_shapes=input_shapes,
                    max_buffer_size=self.replay_buffer_size)

            # Set up target init and update function
            with tf.name_scope("setup_target"):
                actor_init_ops, actor_update_ops = get_target_ops(
                    self.actor.global_vars, target_actor.global_vars, self.tau)
                critic_init_ops, critic_update_ops = get_target_ops(
                    self.critic.global_vars, target_critic.global_vars,
                    self.tau)
                target_init_op = actor_init_ops + critic_init_ops
                target_update_op = actor_update_ops + critic_update_ops

            f_init_target = tensor_utils.compile_function(
                inputs=[], outputs=target_init_op)
            f_update_target = tensor_utils.compile_function(
                inputs=[], outputs=target_update_op)

            with tf.name_scope("inputs"):
                obs_dim = (
                    self.input_dims["observation"] + self.input_dims["goal"]
                ) if self.use_her else self.input_dims["observation"]
                y = tf.placeholder(tf.float32, shape=(None, 1), name="input_y")
                obs = tf.placeholder(
                    tf.float32,
                    shape=(None, obs_dim),
                    name="input_observation")
                actions = tf.placeholder(
                    tf.float32,
                    shape=(None, self.input_dims["action"]),
                    name="input_action")

            # Set up actor training function
            next_action = self.actor.get_action_sym(obs, name="actor_action")
            next_qval = self.critic.get_qval_sym(
                obs, next_action, name="actor_qval")
            with tf.name_scope("action_loss"):
                action_loss = -tf.reduce_mean(next_qval)
                if self.actor_weight_decay > 0.:
                    actor_reg = tc.layers.apply_regularization(
                        tc.layers.l2_regularizer(self.actor_weight_decay),
                        weights_list=self.actor.regularizable_vars)
                    action_loss += actor_reg

            with tf.name_scope("minimize_action_loss"):
                actor_train_op = self.actor_optimizer(
                    self.actor_lr, name="ActorOptimizer").minimize(
                        action_loss, var_list=self.actor.trainable_vars)

            f_train_actor = tensor_utils.compile_function(
                inputs=[obs], outputs=[actor_train_op, action_loss])

            # Set up critic training function
            qval = self.critic.get_qval_sym(obs, actions, name="q_value")
            with tf.name_scope("qval_loss"):
                qval_loss = tf.reduce_mean(tf.squared_difference(y, qval))
                if self.critic_weight_decay > 0.:
                    critic_reg = tc.layers.apply_regularization(
                        tc.layers.l2_regularizer(self.critic_weight_decay),
                        weights_list=self.critic.regularizable_vars)
                    qval_loss += critic_reg

            with tf.name_scope("minimize_critic_loss"):
                critic_train_op = self.critic_optimizer(
                    self.critic_lr, name="CriticOptimizer").minimize(
                        qval_loss, var_list=self.critic.trainable_vars)

            f_train_critic = tensor_utils.compile_function(
                inputs=[y, obs, actions],
                outputs=[critic_train_op, qval_loss, qval])

            self.f_train_actor = f_train_actor
            self.f_train_critic = f_train_critic
            self.f_init_target = f_init_target
            self.f_update_target = f_update_target
            self.replay_buffer = replay_buffer
            self.target_critic = target_critic
            self.target_actor = target_actor

    def _learn(self):
        """
        Perform algorithm optimizing.

        Returns:
            action_loss: Loss of action predicted by the policy network.
            qval_loss: Loss of q value predicted by the q network.
            ys: y_s.
            qval: Q value predicted by the q network.

        """
        if self.use_her:
            transitions = self.replay_buffer.sample(self.batch_size)
            observations = transitions["observation"]
            rewards = transitions["reward"]
            actions = transitions["action"]
            next_observations = transitions["next_observation"]
            goals = transitions["goal"]

            next_inputs = np.concatenate((next_observations, goals), axis=-1)
            inputs = np.concatenate((observations, goals), axis=-1)

            rewards = rewards.reshape(-1, 1)

            target_actions, _ = self.target_actor.get_actions(next_inputs)
            target_qvals = self.target_critic.get_qval(next_inputs,
                                                       target_actions)

            clip_range = (-self.clip_return, 0.
                          if self.clip_pos_returns else np.inf)
            ys = np.clip(rewards + self.discount * target_qvals, clip_range[0],
                         clip_range[1])

            _, qval_loss, qval = self.f_train_critic(ys, inputs, actions)
            _, action_loss = self.f_train_actor(inputs)
        else:
            transitions = self.replay_buffer.sample(self.batch_size)
            observations = transitions["observation"]
            rewards = transitions["reward"]
            actions = transitions["action"]
            terminals = transitions["terminal"]
            next_observations = transitions["next_observation"]

            rewards = rewards.reshape(-1, 1)
            terminals = terminals.reshape(-1, 1)

            target_actions, _ = self.target_actor.get_actions(
                next_observations)
            target_qvals = self.target_critic.get_qval(next_observations,
                                                       target_actions)

            ys = rewards + (1.0 - terminals) * self.discount * target_qvals

            _, qval_loss, qval = self.f_train_critic(ys, observations, actions)
            _, action_loss = self.f_train_actor(observations)
            self.f_update_target()

        return qval_loss, ys, qval, action_loss

    def get_itr_snapshot(self, itr):
        return dict(itr=itr, policy=self.actor, env=self.env)


def get_target_ops(vars, target_vars, tau):
    """Get target network update operations."""
    update_ops = []
    init_ops = []
    assert len(vars) == len(target_vars)
    for var, target_var in zip(vars, target_vars):
        init_ops.append(tf.assign(target_var, var))
        update_ops.append(
            tf.assign(target_var, tau * var + (1.0 - tau) * target_var))
    return init_ops, update_ops
