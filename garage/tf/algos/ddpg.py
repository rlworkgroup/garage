import pickle

import numpy as np
import pyprind
import tensorflow as tf
import tensorflow.contrib as tc

from garage.algos.base import RLAlgorithm
from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.misc import tensor_utils
from garage.tf.plotter import Plotter
from garage.tf.replay_buffer import ReplayBuffer


class DDPG(RLAlgorithm):
    """
    A DDPG model based on https://arxiv.org/pdf/1509.02971.pdf.

    Example:
        $ python rllab/tf/launchers/ddpg_pendulum.py
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
                 critic_optimizer=None):
        """
        Args:
            env(): Environment.
            actor(rllab.tf.policies.ContinuousMLPPolicy): Policy network.
            critic(rllab.tf.q_functions.ContinuousMLPQFunction): Q Value network.
            n_epochs(int, optional): Number of epochs.
            n_epoch_cycles(int, optional): Number of epoch cycles.
            n_rollout_steps(int, optional): Number of rollout steps.
            n_train_steps(int, optional): Number of train steps.
            reward_scale(float): The scaling factor applied to the rewards when
         training.
            batch_size(int): Number of samples for each minibatch.
            target_update_tau(float): Interpolation parameter for doing the soft
         target update.
            discount(float): Discount factor for the cumulative return.
            actor_lr(float): Learning rate for training policy network.
            critic_lr(float): Learning rate for training q value network.
            actor_weight_decay(float): L2 weight decay factor for parameters of
         the policy network.
            critic_weight_decay(float): L2 weight decay factor for parameters of
         the q value network.
            replay_buffer_size(int): Size of the replay buffer.
            min_buffer_size(int): Minimum size of the replay buffer to start training.
            exploration_strategy(): Exploration strategy.
            plot(bool): Whether to visualize the policy performance after each
         eval_interval.
            pause_for_plot(bool): Whether to pause before continuing when plotting.
            actor_optimizer(): Optimizer for training policy network.
            critic_optimizer(): Optimizer for training q function network.
        """
        self.env = env

        self.observation_dim = env.observation_space.shape[-1]
        self.action_dim = env.action_space.shape[-1]
        self.action_bound = env.action_space.high

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
        self.critic_optimzier = critic_optimizer

        self._initialize()

    @overrides
    def train(self, sess=None):
        replay_buffer = self.opt_info["replay_buffer"]
        f_init_target = self.opt_info["f_init_target"]
        created_session = True if (sess is None) else False
        if sess is None:
            sess = tf.Session()
            sess.__enter__()

        # Start plotter
        if self.plot:
            self.plotter = Plotter(self.env, self.actor, sess)
            self.plotter.start()

        sess.run(tf.global_variables_initializer())
        f_init_target()

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
        terminal = False

        for epoch in range(self.n_epochs):
            logger.push_prefix('epoch #%d | ' % epoch)
            logger.log("Training started")
            for epoch_cycle in pyprind.prog_bar(range(self.n_epoch_cycles)):
                for rollout in range(self.n_rollout_steps):
                    action = self.es.get_action(rollout, observation,
                                                self.actor)
                    assert action.shape == self.env.action_space.shape

                    next_observation, reward, terminal, info = self.env.step(
                        action)
                    episode_reward += reward
                    episode_step += 1

                    replay_buffer.add_transition(observation, action,
                                                 reward * self.reward_scale,
                                                 terminal, next_observation)

                    observation = next_observation

                    if terminal:
                        episode_rewards.append(episode_reward)
                        episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        episodes += 1

                        observation = self.env.reset()
                        if self.es:
                            self.es.reset()

                for train_itr in range(self.n_train_steps):
                    if replay_buffer.size >= self.min_buffer_size:
                        critic_loss, y, q, action_loss = self._learn()

                        episode_actor_losses.append(action_loss)
                        episode_critic_losses.append(critic_loss)
                        epoch_ys.append(y)
                        epoch_qs.append(q)

            logger.log("Training finished")
            if replay_buffer.size >= self.min_buffer_size:
                logger.record_tabular('Epoch', epoch)
                logger.record_tabular('Episodes', episodes)
                logger.record_tabular('AverageReturn',
                                      np.mean(episode_rewards))
                logger.record_tabular('StdReturn', np.std(episode_rewards))
                logger.record_tabular('AveragePolicyLoss',
                                      np.mean(episode_actor_losses))
                logger.record_tabular('AverageQFunctionLoss',
                                      np.mean(episode_critic_losses))
                logger.record_tabular('AverageQ', np.mean(epoch_qs))
                logger.record_tabular('MaxQ', np.max(epoch_qs))
                logger.record_tabular('AverageAbsQ', np.mean(np.abs(epoch_qs)))
                logger.record_tabular('AverageY', np.mean(epoch_ys))
                logger.record_tabular('MaxY', np.max(epoch_ys))
                logger.record_tabular('AverageAbsY', np.mean(np.abs(epoch_ys)))

                episode_rewards = []
                episode_actor_losses = []
                episode_critic_losses = []
                epoch_ys = []
                epoch_qs = []

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
        """
        Set up the actor, critic and target network.
        """
        # Set up the actor and critic network
        self.actor.build_net()
        self.critic.build_net()

        # Create target actor and critic network
        target_actor = pickle.loads(pickle.dumps(self.actor))
        target_critic = pickle.loads(pickle.dumps(self.critic))

        # Set up the target network
        target_actor.name = "TargetActor"
        target_actor.build_net(trainable=False)
        target_critic.name = "TargetCritic"
        target_critic.build_net(trainable=False)

        # Initialize replay buffer
        replay_buffer = ReplayBuffer(self.replay_buffer_size,
                                     self.observation_dim, self.action_dim)

        # Set up target init and update function
        actor_init_ops, actor_update_ops = self._get_target_ops(
            self.actor.global_vars, target_actor.global_vars, self.tau)
        critic_init_ops, critic_update_ops = self._get_target_ops(
            self.critic.global_vars, target_critic.global_vars, self.tau)
        target_init_op = actor_init_ops + critic_init_ops
        target_update_op = actor_update_ops + critic_update_ops

        f_init_target = tensor_utils.compile_function(
            inputs=[], outputs=target_init_op)
        f_update_target = tensor_utils.compile_function(
            inputs=[], outputs=target_update_op)

        y = tf.placeholder(tf.float32, shape=(None, 1), name="input_y")
        obs = tf.placeholder(
            tf.float32,
            shape=(None, self.observation_dim),
            name="input_observation")
        actions = tf.placeholder(
            tf.float32, shape=(None, self.action_dim), name="input_action")

        # Set up actor training function
        next_action = self.actor.get_action_sym(obs, name="actor_action")
        next_qval = self.critic.get_qval_sym(
            obs, next_action, name="actor_qval")
        action_loss = -tf.reduce_mean(next_qval)
        if self.actor_weight_decay > 0.:
            actor_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.actor_weight_decay),
                weights_list=self.actor.regularizable_vars)
            action_loss += actor_reg

        actor_train_op = self.actor_optimizer(
            self.actor_lr, name="ActorOptimizer").minimize(
                action_loss, var_list=self.actor.trainable_vars)

        f_train_actor = tensor_utils.compile_function(
            inputs=[obs], outputs=[actor_train_op, action_loss])

        # Set up critic training function
        qval = self.critic.get_qval_sym(obs, actions, name="q_value")
        qval_loss = tf.reduce_mean(tf.squared_difference(y, qval))
        if self.critic_weight_decay > 0.:
            critic_reg = tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self.critic_weight_decay),
                weights_list=self.critic.regularizable_vars)
            qval_loss += critic_reg
        critic_train_op = self.critic_optimzier(
            self.critic_lr, name="CriticOptimizer").minimize(
                qval_loss, var_list=self.critic.trainable_vars)

        f_train_critic = tensor_utils.compile_function(
            inputs=[y, obs, actions],
            outputs=[critic_train_op, qval_loss, qval])

        self.opt_info = dict(
            f_train_actor=f_train_actor,
            f_train_critic=f_train_critic,
            f_init_target=f_init_target,
            f_update_target=f_update_target,
            replay_buffer=replay_buffer,
            target_critic=target_critic,
            target_actor=target_actor)

    def _learn(self):
        """
        Perform algorithm optimizing.

        Returns:
            action_loss: Loss of action predicted by the policy network.
            qval_loss: Loss of q value predicted by the q network.
            ys: y_s.
            qval: Q value predicted by the q network.
        """
        replay_buffer = self.opt_info["replay_buffer"]
        target_actor = self.opt_info["target_actor"]
        target_critic = self.opt_info["target_critic"]
        f_train_critic = self.opt_info["f_train_critic"]
        f_train_actor = self.opt_info["f_train_actor"]
        f_update_target = self.opt_info["f_update_target"]

        transitions = replay_buffer.random_sample(self.batch_size)
        observations = transitions["observations"]
        rewards = transitions["rewards"]
        actions = transitions["actions"]
        terminals = transitions["terminals"]
        next_observations = transitions["next_observations"]

        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        target_actions = target_actor.get_actions(next_observations)
        target_qvals = target_critic.get_qval(next_observations,
                                              target_actions)

        ys = rewards + (1.0 - terminals) * self.discount * target_qvals

        _, qval_loss, qval = f_train_critic(ys, observations, actions)
        _, action_loss = f_train_actor(observations)
        f_update_target()

        return qval_loss, ys, qval, action_loss

    def _get_target_ops(self, vars, target_vars, tau):
        update_ops = []
        init_ops = []
        assert len(vars) == len(target_vars)
        for var, target_var in zip(vars, target_vars):
            init_ops.append(tf.assign(target_var, var))
            update_ops.append(
                tf.assign(target_var, tau * var + (1.0 - tau) * target_var))
        return init_ops, update_ops
