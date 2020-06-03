"""Deep Deterministic Policy Gradient (DDPG) implementation in TensorFlow."""
from collections import deque

from dowel import logger, tabular
import numpy as np
import tensorflow as tf

from garage import _Default, make_optimizer
from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.tf.misc import tensor_utils


class DDPG(OffPolicyRLAlgorithm):
    """A DDPG model based on https://arxiv.org/pdf/1509.02971.pdf.

    DDPG, also known as Deep Deterministic Policy Gradient, uses actor-critic
    method to optimize the policy and reward prediction. It uses a supervised
    method to update the critic network and policy gradient to update the actor
    network. And there are exploration strategy, replay buffer and target
    networks involved to stabilize the training process.

    Example:
        $ python garage/examples/tf/ddpg_pendulum.py

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.Policy): Policy.
        qf (object): The q value network.
        replay_buffer (garage.replay_buffer.ReplayBuffer): Replay buffer.
        steps_per_epoch (int): Number of train_once calls per epoch.
        n_train_steps (int): Training steps.
        max_path_length (int): Maximum path length. The episode will
            terminate when length of trajectory reaches max_path_length.
        max_eval_path_length (int or None): Maximum length of paths used for
            off-policy evaluation. If None, defaults to `max_path_length`.
        buffer_batch_size (int): Batch size of replay buffer.
        min_buffer_size (int): The minimum buffer size for replay buffer.
        rollout_batch_size (int): Roll out batch size.
        exploration_policy (garage.np.exploration_policies.ExplorationPolicy):
            Exploration strategy.
        target_update_tau (float): Interpolation parameter for doing the
            soft target update.
        policy_lr (float): Learning rate for training policy network.
        qf_lr (float): Learning rate for training q value network.
        discount(float): Discount factor for the cumulative return.
        policy_weight_decay (float): L2 regularization factor for parameters
            of the policy network. Value of 0 means no regularization.
        qf_weight_decay (float): L2 regularization factor for parameters
            of the q value network. Value of 0 means no regularization.
        policy_optimizer (tf.Optimizer): Optimizer for training policy network.
        qf_optimizer (tf.Optimizer): Optimizer for training q function
            network.
        clip_pos_returns (bool): Whether or not clip positive returns.
        clip_return (float): Clip return to be in [-clip_return,
            clip_return].
        max_action (float): Maximum action magnitude.
        reward_scale (float): Reward scale.
        smooth_return (bool): Whether to smooth the return.
        name (str): Name of the algorithm shown in computation graph.

    """

    def __init__(
            self,
            env_spec,
            policy,
            qf,
            replay_buffer,
            *,  # Everything after this is numbers.
            steps_per_epoch=20,
            n_train_steps=50,
            max_path_length=None,
            max_eval_path_length=None,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            rollout_batch_size=1,
            exploration_policy=None,
            target_update_tau=0.01,
            discount=0.99,
            policy_weight_decay=0,
            qf_weight_decay=0,
            policy_optimizer=tf.compat.v1.train.AdamOptimizer,
            qf_optimizer=tf.compat.v1.train.AdamOptimizer,
            policy_lr=_Default(1e-4),
            qf_lr=_Default(1e-3),
            clip_pos_returns=False,
            clip_return=np.inf,
            max_action=None,
            reward_scale=1.,
            smooth_return=True,
            name='DDPG'):
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
        self._policy_optimizer = policy_optimizer
        self._qf_optimizer = qf_optimizer
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr

        super(DDPG, self).__init__(env_spec=env_spec,
                                   policy=policy,
                                   qf=qf,
                                   n_train_steps=n_train_steps,
                                   steps_per_epoch=steps_per_epoch,
                                   max_path_length=max_path_length,
                                   max_eval_path_length=max_eval_path_length,
                                   buffer_batch_size=buffer_batch_size,
                                   min_buffer_size=min_buffer_size,
                                   rollout_batch_size=rollout_batch_size,
                                   exploration_policy=exploration_policy,
                                   replay_buffer=replay_buffer,
                                   use_target=True,
                                   discount=discount,
                                   reward_scale=reward_scale,
                                   smooth_return=smooth_return)

    def init_opt(self):
        """Build the loss function and init the optimizer."""
        with tf.name_scope(self._name):
            # Create target policy and qf network
            self.target_policy_f_prob_online = tensor_utils.compile_function(
                inputs=[self._target_policy.model.networks['default'].input],
                outputs=self._target_policy.model.networks['default'].outputs)
            self.target_qf_f_prob_online = tensor_utils.compile_function(
                inputs=self._target_qf.model.networks['default'].inputs,
                outputs=self._target_qf.model.networks['default'].outputs)

            # Set up target init and update function
            with tf.name_scope('setup_target'):
                ops = tensor_utils.get_target_ops(
                    self.policy.get_global_vars(),
                    self._target_policy.get_global_vars(), self._tau)
                policy_init_ops, policy_update_ops = ops
                qf_init_ops, qf_update_ops = tensor_utils.get_target_ops(
                    self.qf.get_global_vars(),
                    self._target_qf.get_global_vars(), self._tau)
                target_init_op = policy_init_ops + qf_init_ops
                target_update_op = policy_update_ops + qf_update_ops

            f_init_target = tensor_utils.compile_function(
                inputs=[], outputs=target_init_op)
            f_update_target = tensor_utils.compile_function(
                inputs=[], outputs=target_update_op)

            with tf.name_scope('inputs'):
                obs_dim = self.env_spec.observation_space.flat_dim
                input_y = tf.compat.v1.placeholder(tf.float32,
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
                if self._policy_weight_decay > 0.:
                    regularizer = tf.keras.regularizers.l2(
                        self._policy_weight_decay)
                    for var in self.policy.get_regularizable_vars():
                        policy_reg = regularizer(var)
                        action_loss += policy_reg

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
            with tf.name_scope('qval_loss'):
                qval_loss = tf.reduce_mean(
                    tf.compat.v1.squared_difference(input_y, qval))
                if self._qf_weight_decay > 0.:
                    regularizer = tf.keras.regularizers.l2(
                        self._qf_weight_decay)
                    for var in self.qf.get_regularizable_vars():
                        qf_reg = regularizer(var)
                        qval_loss += qf_reg

            with tf.name_scope('minimize_qf_loss'):
                qf_optimizer = make_optimizer(self._qf_optimizer,
                                              learning_rate=self._qf_lr,
                                              name='QFunctionOptimizer')
                qf_train_op = qf_optimizer.minimize(
                    qval_loss, var_list=self.qf.get_trainable_vars())

            f_train_qf = tensor_utils.compile_function(
                inputs=[input_y, obs, actions],
                outputs=[qf_train_op, qval_loss, qval])

            self.f_train_policy = f_train_policy
            self.f_train_qf = f_train_qf
            self.f_init_target = f_init_target
            self.f_update_target = f_update_target

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        data = self.__dict__.copy()
        del data['target_policy_f_prob_online']
        del data['target_qf_f_prob_online']
        del data['f_train_policy']
        del data['f_train_qf']
        del data['f_init_target']
        del data['f_update_target']
        return data

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

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
                qf_loss, y_s, qval, policy_loss = self.optimize_policy()

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

    def optimize_policy(self):
        """Perform algorithm optimizing.

        Returns:
            float: Loss of action predicted by the policy network
            float: Loss of q value predicted by the q network.
            float: ys.
            float: Q value predicted by the q network.

        """
        transitions = self.replay_buffer.sample_transitions(
            self.buffer_batch_size)

        observations = transitions['observations']
        next_observations = transitions['next_observations']
        rewards = transitions['rewards'].reshape(-1, 1)
        actions = transitions['actions']
        terminals = transitions['terminals'].reshape(-1, 1)

        next_inputs = next_observations
        inputs = observations

        target_actions = self.target_policy_f_prob_online(next_inputs)
        target_qvals = self.target_qf_f_prob_online(next_inputs,
                                                    target_actions)

        clip_range = (-self._clip_return,
                      0. if self._clip_pos_returns else self._clip_return)
        ys = np.clip(
            rewards + (1.0 - terminals) * self.discount * target_qvals,
            clip_range[0], clip_range[1])

        _, qval_loss, qval = self.f_train_qf(ys, inputs, actions)
        _, action_loss = self.f_train_policy(inputs)

        self.f_update_target()

        return qval_loss, ys, qval, action_loss
