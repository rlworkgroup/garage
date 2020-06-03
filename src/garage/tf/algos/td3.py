"""This module implements a TD3 model.

TD3, or Twin Delayed Deep Deterministic Policy Gradient, uses actor-critic
method to optimize the policy and reward prediction. Notably, it uses the
minimum value of two critics instead of one to limit overestimation.
"""
from collections import deque

from dowel import logger, tabular
import numpy as np
import tensorflow as tf

from garage import _Default, make_optimizer
from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.tf.misc import tensor_utils


class TD3(OffPolicyRLAlgorithm):
    """Implementation of TD3.

    Based on https://arxiv.org/pdf/1802.09477.pdf.

    Example:
        $ python garage/examples/tf/td3_pendulum.py

    Args:
        env_spec (garage.envs.EnvSpec): Environment.
        policy (garage.tf.policies.Policy): Policy.
        qf (garage.tf.q_functions.QFunction): Q-function.
        qf2 (garage.tf.q_functions.QFunction): Q function to use
        replay_buffer (garage.replay_buffer.ReplayBuffer): Replay buffer.
        target_update_tau (float): Interpolation parameter for doing the
            soft target update.
        policy_lr (float): Learning rate for training policy network.
        qf_lr (float): Learning rate for training q value network.
        policy_weight_decay (float): L2 weight decay factor for parameters
            of the policy network.
        qf_weight_decay (float): L2 weight decay factor for parameters
            of the q value network.
        policy_optimizer (tf.python.training.optimizer.Optimizer):
            Optimizer for training policy network.
        qf_optimizer (tf.python.training.optimizer.Optimizer):
            Optimizer for training q function network.
        clip_pos_returns (boolean): Whether or not clip positive returns.
        clip_return (float): Clip return to be in [-clip_return,
            clip_return].
        discount (float): Discount factor for the cumulative return.
        max_action (float): Maximum action magnitude.
        name (str): Name of the algorithm shown in computation graph.
        steps_per_epoch (int): Number of batches of samples in each epoch.
        max_path_length (int): Maximum length of a path.
        max_eval_path_length (int or None): Maximum length of paths used for
            off-policy evaluation. If None, defaults to `max_path_length`.
        n_train_steps (int): Number of optimizations in each epoch cycle.
        buffer_batch_size (int): Size of replay buffer.
        min_buffer_size (int):
            Number of samples in replay buffer before first optimization.
        rollout_batch_size (int): Roll out batch size.
        reward_scale (float): Scale to reward.
        exploration_policy_sigma (float): Action noise sigma.
        exploration_policy_clip (float): Action noise clip.
        actor_update_period (int): Action update period.
        smooth_return (bool):
            If True, do statistics on all samples collection.
            Otherwise do statistics on one batch.
        exploration_policy (garage.np.exploration_policies.ExplorationPolicy):
            Exploration strategy.

    """

    def __init__(
            self,
            env_spec,
            policy,
            qf,
            qf2,
            replay_buffer,
            *,  # Everything after this is numbers.
            target_update_tau=0.01,
            policy_weight_decay=0,
            qf_weight_decay=0,
            policy_optimizer=tf.compat.v1.train.AdamOptimizer,
            qf_optimizer=tf.compat.v1.train.AdamOptimizer,
            policy_lr=_Default(1e-4),
            qf_lr=_Default(1e-3),
            clip_pos_returns=False,
            clip_return=np.inf,
            discount=0.99,
            max_action=None,
            name='TD3',
            steps_per_epoch=20,
            max_path_length=None,
            max_eval_path_length=None,
            n_train_steps=50,
            buffer_batch_size=64,
            min_buffer_size=1e4,
            rollout_batch_size=1,
            reward_scale=1.,
            exploration_policy_sigma=0.2,
            actor_update_period=2,
            exploration_policy_clip=0.5,
            smooth_return=True,
            exploration_policy=None):
        action_bound = env_spec.action_space.high
        self._max_action = action_bound if max_action is None else max_action
        self._tau = target_update_tau
        self._policy_weight_decay = policy_weight_decay
        self._qf_weight_decay = qf_weight_decay
        self._name = name
        self._clip_pos_returns = clip_pos_returns
        self._clip_return = clip_return
        self._success_history = deque(maxlen=100)

        self._episode_rewards = []
        self._episode_policy_losses = []
        self._episode_qf_losses = []
        self._epoch_ys = []
        self._epoch_qs = []

        self._target_policy = policy.clone('target_policy')
        self._target_qf = qf.clone('target_qf')

        self.qf2 = qf2
        self._exploration_policy_sigma = exploration_policy_sigma
        self._exploration_policy_clip = exploration_policy_clip
        self._actor_update_period = actor_update_period
        self._action_loss = None

        self._target_qf2 = qf2.clone('target_qf2')
        self._policy_optimizer = policy_optimizer
        self._qf_optimizer = qf_optimizer
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr

        super(TD3, self).__init__(env_spec=env_spec,
                                  policy=policy,
                                  qf=qf,
                                  replay_buffer=replay_buffer,
                                  discount=discount,
                                  steps_per_epoch=steps_per_epoch,
                                  max_path_length=max_path_length,
                                  max_eval_path_length=max_eval_path_length,
                                  n_train_steps=n_train_steps,
                                  buffer_batch_size=buffer_batch_size,
                                  min_buffer_size=min_buffer_size,
                                  rollout_batch_size=rollout_batch_size,
                                  reward_scale=reward_scale,
                                  smooth_return=smooth_return,
                                  exploration_policy=exploration_policy)

    def init_opt(self):
        """Build the loss function and init the optimizer."""
        with tf.name_scope(self._name):
            # Create target policy (actor) and qf (critic) networks
            self.target_policy_f_prob_online = tensor_utils.compile_function(
                inputs=[self._target_policy.model.networks['default'].input],
                outputs=self._target_policy.model.networks['default'].outputs)

            self.target_qf_f_prob_online = tensor_utils.compile_function(
                inputs=self._target_qf.model.networks['default'].inputs,
                outputs=self._target_qf.model.networks['default'].outputs)

            self.target_qf2_f_prob_online = tensor_utils.compile_function(
                inputs=self._target_qf2.model.networks['default'].inputs,
                outputs=self._target_qf2.model.networks['default'].outputs)

            # Set up target init and update functions
            with tf.name_scope('setup_target'):
                policy_init_op, policy_update_op = tensor_utils.get_target_ops(
                    self.policy.get_global_vars(),
                    self._target_policy.get_global_vars(), self._tau)
                qf_init_ops, qf_update_ops = tensor_utils.get_target_ops(
                    self.qf.get_global_vars(),
                    self._target_qf.get_global_vars(), self._tau)
                qf2_init_ops, qf2_update_ops = tensor_utils.get_target_ops(
                    self.qf2.get_global_vars(),
                    self._target_qf2.get_global_vars(), self._tau)
                target_init_op = policy_init_op + qf_init_ops + qf2_init_ops
                target_update_op = (policy_update_op + qf_update_ops +
                                    qf2_update_ops)

            f_init_target = tensor_utils.compile_function(
                inputs=[], outputs=target_init_op)
            f_update_target = tensor_utils.compile_function(
                inputs=[], outputs=target_update_op)

            with tf.name_scope('inputs'):
                obs_dim = self.env_spec.observation_space.flat_dim
                y = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, 1),
                                             name='input_y')
                obs = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, obs_dim),
                                               name='input_observation')
                actions = tf.compat.v1.placeholder(
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
                policy_optimizer = make_optimizer(
                    self._policy_optimizer,
                    learning_rate=self._policy_lr,
                    name='PolicyOptimizer')
                policy_train_op = policy_optimizer.minimize(
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
                qf_optimizer = make_optimizer(self._qf_optimizer,
                                              learning_rate=self._qf_lr,
                                              name='QFunctionOptimizer')
                qf_train_op = qf_optimizer.minimize(
                    qval1_loss, var_list=self.qf.get_trainable_vars())
                qf2_train_op = qf_optimizer.minimize(
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
        """Object.__getstate__.

        Returns:
            dict: State dictionary.

        """
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
        """Object.__setstate__.

        Args:
            state (dict): Current state.

        """
        self.__dict__.update(state)
        self.init_opt()

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            np.float64: Average return.

        """
        paths = self.process_samples(itr, paths)

        epoch = itr / self.steps_per_epoch

        self._episode_rewards.extend([
            path for path, complete in zip(paths['undiscounted_returns'],
                                           paths['complete']) if complete
        ])
        self._success_history.extend([
            path for path, complete in zip(paths['success_history'],
                                           paths['complete']) if complete
        ])

        # Avoid calculating the mean of an empty list in cases where
        # all paths were non-terminal.

        last_average_return = np.NaN
        avg_success_rate = 0

        if self._episode_rewards:
            last_average_return = np.mean(self._episode_rewards)

        if self._success_history:
            if itr % self.steps_per_epoch == 0 and self._buffer_prefilled:
                avg_success_rate = np.mean(self._success_history)

        self.log_diagnostics(paths)
        for _ in range(self.n_train_steps):
            if self._buffer_prefilled:
                qf_loss, y_s, qval, policy_loss = self.optimize_policy(itr)

                self._episode_policy_losses.append(policy_loss)
                self._episode_qf_losses.append(qf_loss)
                self._epoch_ys.append(y_s)
                self._epoch_qs.append(qval)

        if itr % self.steps_per_epoch == 0:
            logger.log('Training finished')

            if self._buffer_prefilled:
                tabular.record('Epoch', epoch)
                tabular.record('Policy/AveragePolicyLoss',
                               np.mean(self._episode_policy_losses))
                tabular.record('QFunction/AverageQFunctionLoss',
                               np.mean(self._episode_qf_losses))
                tabular.record('QFunction/AverageQ', np.mean(self._epoch_qs))
                tabular.record('QFunction/MaxQ', np.max(self._epoch_qs))
                tabular.record('QFunction/AverageAbsQ',
                               np.mean(np.abs(self._epoch_qs)))
                tabular.record('QFunction/AverageY', np.mean(self._epoch_ys))
                tabular.record('QFunction/MaxY', np.max(self._epoch_ys))
                tabular.record('QFunction/AverageAbsY',
                               np.mean(np.abs(self._epoch_ys)))
                tabular.record('AverageSuccessRate', avg_success_rate)

            if not self.smooth_return:
                self._episode_rewards = []
                self._episode_policy_losses = []
                self._episode_qf_losses = []
                self._epoch_ys = []
                self._epoch_qs = []

            self._success_history.clear()
        return last_average_return

    def optimize_policy(self, itr):
        """Perform algorithm optimizing.

        Args:
            itr(int): Iterations.

        Returns:
            action_loss(float): Loss of action predicted by the policy network.
            qval_loss(float): Loss of q value predicted by the q network.
            ys(float): y_s.
            qval(float): Q value predicted by the q network.

        """
        transitions = self.replay_buffer.sample_transitions(
            self.buffer_batch_size)

        observations = transitions['observations']
        rewards = transitions['rewards']
        actions = transitions['actions']
        next_observations = transitions['next_observations']
        terminals = transitions['terminals']

        next_inputs = next_observations
        inputs = observations

        target_actions = self.target_policy_f_prob_online(next_inputs)

        noise = np.random.normal(0.0, self._exploration_policy_sigma,
                                 target_actions.shape)
        noise = np.clip(noise, -self._exploration_policy_clip,
                        self._exploration_policy_clip)
        target_actions += noise

        target_qvals = self.target_qf_f_prob_online(next_inputs,
                                                    target_actions)
        target_q2vals = self.target_qf2_f_prob_online(next_inputs,
                                                      target_actions)
        target_qvals = np.minimum(target_qvals, target_q2vals)
        ys = (rewards + (1.0 - terminals) * self.discount * target_qvals)

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
