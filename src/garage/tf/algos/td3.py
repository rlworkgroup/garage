"""
This module implements a TD3 model.

TD3, or Twin Delayed Deep Deterministic Policy Gradient, uses actor-critic
method to optimize the policy and reward prediction. Notably, it uses the
minimum value of two critics instead of one to limit overestimation.
"""

import numpy as np
import tensorflow as tf

from garage.tf.algos import DDPG
from garage.tf.misc import tensor_utils


class TD3(DDPG):
    """Implementation of TD3.

    Based on https://arxiv.org/pdf/1802.09477.pdf.

    Example:
        $ python garage/examples/tf/td3_pendulum.py

    Args:
        env_spec(garage.envs.EnvSpec): Environment.
        policy(garage.tf.policies.base.Policy): Policy.
        qf(garage.tf.q_functions.QFunction): Q-function.
        qf2(garage.tf.q_functions.QFunction): Q function to use
        target_update_tau(float): Interpolation parameter for doing the
            soft target update.
        policy_lr(float): Learning rate for training policy network.
        qf_lr(float): Learning rate for training q value network.
        policy_weight_decay(float): L2 weight decay factor for parameters
            of the policy network.
        qf_weight_decay(float): L2 weight decay factor for parameters
            of the q value network.
        policy_optimizer(tf.python.training.optimizer.Optimizer):
            Optimizer for training policy network.
        qf_optimizer(tf.python.training.optimizer.Optimizer):
            Optimizer for training q function network.
        clip_pos_returns(boolean): Whether or not clip positive returns.
        clip_return(float): Clip return to be in [-clip_return,
            clip_return].
        discount(float): Discount factor for the cumulative return.
        max_action(float): Maximum action magnitude.
        name(str): Name of the algorithm shown in computation graph.
        n_epoch_cycles(int): Number of batches of samples in each epoch.
        max_path_length(int): Maximum length of a path.
        n_train_steps(int): Number of optimizations in each epoch cycle.
        buffer_batch_size(int): Size of replay buffer.
        min_buffer_size(int):
            Number of samples in replay buffer before first optimization.
        rollout_batch_size(int):
        reward_scale(float): Scale to reward.
        input_include_goal(bool):
            True if the environment entails a goal in observation.
        smooth_return(bool):
            If True, do statistics on all samples collection.
            Otherwise do statistics on one batch.
        exploration_strategy(
            garage.np.exploration_strategies.ExplorationStrategy):
            Exploration strategy.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 qf,
                 qf2,
                 replay_buffer,
                 target_update_tau=0.01,
                 policy_lr=1e-4,
                 qf_lr=1e-3,
                 policy_weight_decay=0,
                 qf_weight_decay=0,
                 policy_optimizer=tf.compat.v1.train.AdamOptimizer,
                 qf_optimizer=tf.compat.v1.train.AdamOptimizer,
                 clip_pos_returns=False,
                 clip_return=np.inf,
                 discount=0.99,
                 max_action=None,
                 name=None,
                 n_epoch_cycles=20,
                 max_path_length=None,
                 n_train_steps=50,
                 buffer_batch_size=64,
                 min_buffer_size=1e4,
                 rollout_batch_size=1,
                 reward_scale=1.,
                 action_noise_sigma=0.2,
                 actor_update_period=2,
                 action_noise_clip=0.5,
                 input_include_goal=False,
                 smooth_return=True,
                 exploration_strategy=None):
        self.qf2 = qf2
        self._action_noise_sigma = action_noise_sigma
        self._action_noise_clip = action_noise_clip
        self._actor_update_period = actor_update_period
        self._action_loss = None

        self.target_qf2 = qf2.clone('target_qf2')

        super(TD3, self).__init__(env_spec=env_spec,
                                  policy=policy,
                                  qf=qf,
                                  replay_buffer=replay_buffer,
                                  target_update_tau=target_update_tau,
                                  policy_lr=policy_lr,
                                  qf_lr=qf_lr,
                                  policy_weight_decay=policy_weight_decay,
                                  qf_weight_decay=qf_weight_decay,
                                  policy_optimizer=policy_optimizer,
                                  qf_optimizer=qf_optimizer,
                                  clip_pos_returns=clip_pos_returns,
                                  clip_return=clip_return,
                                  discount=discount,
                                  max_action=max_action,
                                  name=name,
                                  n_epoch_cycles=n_epoch_cycles,
                                  max_path_length=max_path_length,
                                  n_train_steps=n_train_steps,
                                  buffer_batch_size=buffer_batch_size,
                                  min_buffer_size=min_buffer_size,
                                  rollout_batch_size=rollout_batch_size,
                                  reward_scale=reward_scale,
                                  input_include_goal=input_include_goal,
                                  smooth_return=smooth_return,
                                  exploration_strategy=exploration_strategy)

    def init_opt(self):
        """Build the loss function and init the optimizer."""
        with tf.name_scope(self.name, 'TD3'):
            # Create target policy (actor) and qf (critic) networks
            self.target_policy_f_prob_online = tensor_utils.compile_function(
                inputs=[self.target_policy.model.networks['default'].input],
                outputs=self.target_policy.model.networks['default'].outputs)

            self.target_qf_f_prob_online = tensor_utils.compile_function(
                inputs=self.target_qf.model.networks['default'].inputs,
                outputs=self.target_qf.model.networks['default'].outputs)

            self.target_qf2_f_prob_online = tensor_utils.compile_function(
                inputs=self.target_qf2.model.networks['default'].inputs,
                outputs=self.target_qf2.model.networks['default'].outputs)

            # Set up target init and update functions
            with tf.name_scope('setup_target'):
                policy_init_op, policy_update_op = tensor_utils.get_target_ops(
                    self.policy.get_global_vars(),
                    self.target_policy.get_global_vars(), self.tau)
                qf_init_ops, qf_update_ops = tensor_utils.get_target_ops(
                    self.qf.get_global_vars(),
                    self.target_qf.get_global_vars(), self.tau)
                qf2_init_ops, qf2_update_ops = tensor_utils.get_target_ops(
                    self.qf2.get_global_vars(),
                    self.target_qf2.get_global_vars(), self.tau)
                target_init_op = policy_init_op + qf_init_ops + qf2_init_ops
                target_update_op = (policy_update_op + qf_update_ops +
                                    qf2_update_ops)

            f_init_target = tensor_utils.compile_function(
                inputs=[], outputs=target_init_op)
            f_update_target = tensor_utils.compile_function(
                inputs=[], outputs=target_update_op)

            with tf.name_scope('inputs'):
                if self.input_include_goal:
                    obs_dim = self.env_spec.observation_space.\
                        flat_dim_with_keys(['observation', 'desired_goal'])
                else:
                    obs_dim = self.env_spec.observation_space.flat_dim
                y = tf.placeholder(tf.float32, shape=(None, 1), name='input_y')
                obs = tf.placeholder(tf.float32,
                                     shape=(None, obs_dim),
                                     name='input_observation')
                actions = tf.placeholder(
                    tf.float32,
                    shape=(None, self.env_spec.action_space.flat_dim),
                    name='input_action')

            # Set up policy training function
            next_action = self.policy.get_action_sym(obs, name='policy_action')
            next_qval = self.qf.get_qval_sym(obs,
                                             next_action,
                                             name='policy_action_qval')
            with tf.name_scope('action_loss'):
                action_loss = -tf.reduce_mean(next_qval)

            with tf.name_scope('minimize_action_loss'):
                policy_train_op = self.policy_optimizer(
                    self.policy_lr, name='PolicyOptimizer').minimize(
                        action_loss, var_list=self.policy.get_trainable_vars())

            f_train_policy = tensor_utils.compile_function(
                inputs=[obs], outputs=[policy_train_op, action_loss])

            # Set up qf training function
            qval = self.qf.get_qval_sym(obs, actions, name='q_value')
            q2val = self.qf2.get_qval_sym(obs, actions, name='q2_value')
            with tf.name_scope('qval1_loss'):
                qval1_loss = tf.reduce_mean(tf.math.squared_difference(
                    y, qval))
            with tf.name_scope('qval2_loss'):
                qval2_loss = tf.reduce_mean(
                    tf.math.squared_difference(y, q2val))

            with tf.name_scope('minimize_qf_loss'):
                qf_train_op = self.qf_optimizer(
                    self.qf_lr, name='QFunctionOptimizer').minimize(
                        qval1_loss, var_list=self.qf.get_trainable_vars())
                qf2_train_op = self.qf_optimizer(
                    self.qf_lr, name='QFunctionOptimizer').minimize(
                        qval2_loss, var_list=self.qf2.get_trainable_vars())

            f_train_qf = tensor_utils.compile_function(
                inputs=[y, obs, actions],
                outputs=[qf_train_op, qval1_loss, qval])
            f_train_qf2 = tensor_utils.compile_function(
                inputs=[y, obs, actions],
                outputs=[qf2_train_op, qval2_loss, q2val])

            self.f_train_policy = f_train_policy
            self.f_train_qf = f_train_qf
            self.f_init_target = f_init_target
            self.f_update_target = f_update_target
            self.f_train_qf2 = f_train_qf2

    def __getstate__(self):
        """Object.__getstate__."""
        data = self.__dict__.copy()
        del data['target_policy_f_prob_online']
        del data['target_qf_f_prob_online']
        del data['target_qf2_f_prob_online']
        del data['f_train_policy']
        del data['f_train_qf']
        del data['f_train_qf2']
        del data['f_init_target']
        del data['f_update_target']
        return data

    def __setstate__(self, state):
        """Object.__setstate__."""
        self.__dict__ = state
        self.init_opt()

    def optimize_policy(self, itr, samples_data):
        """Perform algorithm optimizing.

        Args:
            itr(int): Iterations.
            samples_data(list): Processed batch data.

        Returns:
            action_loss(float): Loss of action predicted by the policy network.
            qval_loss(float): Loss of q value predicted by the q network.
            ys(float): y_s.
            qval(float): Q value predicted by the q network.

        """
        transitions = self.replay_buffer.sample(self.buffer_batch_size)
        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        terminals = transitions['terminal']

        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        if self.input_include_goal:
            goals = transitions['goal']
            next_inputs = np.concatenate((next_observations, goals), axis=-1)
            inputs = np.concatenate((observations, goals), axis=-1)
        else:
            next_inputs = next_observations
            inputs = observations

        target_actions = self.target_policy_f_prob_online(next_inputs)

        noise = np.random.normal(0.0, self._action_noise_sigma,
                                 target_actions.shape)
        noise = np.clip(noise, -self._action_noise_clip,
                        self._action_noise_clip)
        target_actions += noise

        target_qvals = self.target_qf_f_prob_online(next_inputs,
                                                    target_actions)
        target_q2vals = self.target_qf2_f_prob_online(next_inputs,
                                                      target_actions)
        target_qvals = np.minimum(target_qvals, target_q2vals)
        ys = (self.reward_scale * rewards +
              (1.0 - terminals) * self.discount * target_qvals)

        _, qval_loss, qval = self.f_train_qf(ys, inputs, actions)
        _, q2val_loss, q2val = self.f_train_qf2(ys, inputs, actions)

        if qval_loss > q2val_loss:
            qval_loss = q2val_loss
            qval = q2val

        # update policy and target networks less frequently
        if self._action_loss is None or (itr % self._actor_update_period) == 0:
            _, self._action_loss = self.f_train_policy(inputs)
            self.f_update_target()

        return qval_loss, ys, qval, self._action_loss
