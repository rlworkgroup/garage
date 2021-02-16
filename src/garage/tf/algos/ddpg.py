"""Deep Deterministic Policy Gradient (DDPG) implementation in TensorFlow."""
# yapf: disable
from dowel import logger, tabular
import numpy as np
import tensorflow as tf

from garage import (_Default, log_performance, make_optimizer,
                    obtain_evaluation_episodes)
from garage.np.algos import RLAlgorithm
from garage.tf import compile_function, get_target_ops

# yapf: enable


class DDPG(RLAlgorithm):
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
        policy (Policy): Policy.
        qf (object): The q value network.
        replay_buffer (ReplayBuffer): Replay buffer.
        sampler (garage.sampler.Sampler): Sampler.
        steps_per_epoch (int): Number of train_once calls per epoch.
        n_train_steps (int): Training steps.
        buffer_batch_size (int): Batch size of replay buffer.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If None, defaults to
            `env_spec.max_episode_length`.
        min_buffer_size (int): The minimum buffer size for replay buffer.
        exploration_policy (ExplorationPolicy): Exploration strategy.
        target_update_tau (float): Interpolation parameter for doing the
            soft target update.
        policy_lr (float): Learning rate for training policy network.
        qf_lr (float): Learning rate for training q value network.
        discount(float): Discount factor for the cumulative return.
        policy_weight_decay (float): L2 regularization factor for parameters
            of the policy network. Value of 0 means no regularization.
        qf_weight_decay (float): L2 regularization factor for parameters
            of the q value network. Value of 0 means no regularization.
        policy_optimizer (tf.compat.v1.train.Optimizer): Optimizer for training
            policy network.
        qf_optimizer (tf.compat.v1.train.Optimizer): Optimizer for training
            Q-function network.
        clip_pos_returns (bool): Whether or not clip positive returns.
        clip_return (float): Clip return to be in [-clip_return,
            clip_return].
        max_action (float): Maximum action magnitude.
        reward_scale (float): Reward scale.
        name (str): Name of the algorithm shown in computation graph.

    """

    def __init__(
            self,
            env_spec,
            policy,
            qf,
            replay_buffer,
            sampler,
            *,  # Everything after this is numbers.
            steps_per_epoch=20,
            n_train_steps=50,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            max_episode_length_eval=None,
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
            name='DDPG'):
        action_bound = env_spec.action_space.high
        self._max_action = action_bound if max_action is None else max_action
        self._tau = target_update_tau
        self._policy_weight_decay = policy_weight_decay
        self._qf_weight_decay = qf_weight_decay
        self._name = name
        self._clip_pos_returns = clip_pos_returns
        self._clip_return = clip_return

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

        self._min_buffer_size = min_buffer_size
        self._qf = qf
        self._steps_per_epoch = steps_per_epoch
        self._n_train_steps = n_train_steps
        self._buffer_batch_size = buffer_batch_size
        self._discount = discount
        self._reward_scale = reward_scale
        self.max_episode_length = env_spec.max_episode_length
        self._max_episode_length_eval = max_episode_length_eval

        if max_episode_length_eval is None:
            self._max_episode_length_eval = env_spec.max_episode_length

        self._eval_env = None

        self._env_spec = env_spec
        self._replay_buffer = replay_buffer
        self.policy = policy
        self.exploration_policy = exploration_policy

        self._sampler = sampler

        self._init_opt()

    # pylint: disable=too-many-statements
    def _init_opt(self):
        """Build the loss function and init the optimizer."""
        with tf.name_scope(self._name):
            # Create target policy and qf network
            with tf.name_scope('inputs'):
                obs_dim = self._env_spec.observation_space.flat_dim
                input_y = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, 1),
                                                   name='input_y')
                obs = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, obs_dim),
                                               name='input_observation')
                actions = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None, self._env_spec.action_space.flat_dim),
                    name='input_action')

            policy_network_outputs = self._target_policy.build(obs,
                                                               name='policy')
            target_qf_outputs = self._target_qf.build(obs, actions, name='qf')

            self._target_policy_f_prob_online = compile_function(
                inputs=[obs], outputs=policy_network_outputs)
            self._target_qf_f_prob_online = compile_function(
                inputs=[obs, actions], outputs=target_qf_outputs)

            # Set up target init and update function
            with tf.name_scope('setup_target'):
                ops = get_target_ops(self.policy.get_global_vars(),
                                     self._target_policy.get_global_vars(),
                                     self._tau)
                policy_init_ops, policy_update_ops = ops
                qf_init_ops, qf_update_ops = get_target_ops(
                    self._qf.get_global_vars(),
                    self._target_qf.get_global_vars(), self._tau)
                target_init_op = policy_init_ops + qf_init_ops
                target_update_op = policy_update_ops + qf_update_ops

            f_init_target = compile_function(inputs=[], outputs=target_init_op)
            f_update_target = compile_function(inputs=[],
                                               outputs=target_update_op)

            with tf.name_scope('inputs'):
                obs_dim = self._env_spec.observation_space.flat_dim
                input_y = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, 1),
                                                   name='input_y')
                obs = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, obs_dim),
                                               name='input_observation')
                actions = tf.compat.v1.placeholder(
                    tf.float32,
                    shape=(None, self._env_spec.action_space.flat_dim),
                    name='input_action')
            # Set up policy training function
            next_action = self.policy.build(obs, name='policy_action')
            next_qval = self._qf.build(obs,
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

            f_train_policy = compile_function(
                inputs=[obs], outputs=[policy_train_op, action_loss])

            # Set up qf training function
            qval = self._qf.build(obs, actions, name='q_value')
            with tf.name_scope('qval_loss'):
                qval_loss = tf.reduce_mean(
                    tf.compat.v1.squared_difference(input_y, qval))
                if self._qf_weight_decay > 0.:
                    regularizer = tf.keras.regularizers.l2(
                        self._qf_weight_decay)
                    for var in self._qf.get_regularizable_vars():
                        qf_reg = regularizer(var)
                        qval_loss += qf_reg

            with tf.name_scope('minimize_qf_loss'):
                qf_optimizer = make_optimizer(self._qf_optimizer,
                                              learning_rate=self._qf_lr,
                                              name='QFunctionOptimizer')
                qf_train_op = qf_optimizer.minimize(
                    qval_loss, var_list=self._qf.get_trainable_vars())

            f_train_qf = compile_function(
                inputs=[input_y, obs, actions],
                outputs=[qf_train_op, qval_loss, qval])

            self._f_train_policy = f_train_policy
            self._f_train_qf = f_train_qf
            self._f_init_target = f_init_target
            self._f_update_target = f_update_target

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        data = self.__dict__.copy()
        del data['_target_policy_f_prob_online']
        del data['_target_qf_f_prob_online']
        del data['_f_train_policy']
        del data['_f_train_qf']
        del data['_f_init_target']
        del data['_f_update_target']
        return data

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        self.__dict__.update(state)
        self._init_opt()

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_returns = [float('nan')]
        trainer.enable_logging = False

        for _ in trainer.step_epochs():
            for cycle in range(self._steps_per_epoch):
                trainer.step_episode = trainer.obtain_episodes(
                    trainer.step_itr)
                if hasattr(self.exploration_policy, 'update'):
                    self.exploration_policy.update(trainer.step_episode)
                self._train_once(trainer.step_itr, trainer.step_episode)
                if (cycle == 0 and self._replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    trainer.enable_logging = True
                    eval_episodes = obtain_evaluation_episodes(
                        self.policy, self._eval_env)
                    last_returns = log_performance(trainer.step_itr,
                                                   eval_episodes,
                                                   discount=self._discount)
                trainer.step_itr += 1

        return np.mean(last_returns)

    def _train_once(self, itr, episodes):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.

        """
        self._replay_buffer.add_episode_batch(episodes)
        epoch = itr / self._steps_per_epoch

        for _ in range(self._n_train_steps):
            if (self._replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                qf_loss, y_s, qval, policy_loss = self._optimize_policy()

                self._episode_policy_losses.append(policy_loss)
                self._episode_qf_losses.append(qf_loss)
                self._epoch_ys.append(y_s)
                self._epoch_qs.append(qval)

        if itr % self._steps_per_epoch == 0:
            logger.log('Training finished')

            if (self._replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
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

    def _optimize_policy(self):
        """Perform algorithm optimizing.

        Returns:
            float: Loss of action predicted by the policy network
            float: Loss of q value predicted by the q network.
            float: ys.
            float: Q value predicted by the q network.

        """
        timesteps = self._replay_buffer.sample_timesteps(
            self._buffer_batch_size)

        observations = timesteps.observations
        rewards = timesteps.rewards.reshape(-1, 1)
        actions = timesteps.actions
        next_observations = timesteps.next_observations
        terminals = timesteps.terminals.reshape(-1, 1)

        rewards *= self._reward_scale

        next_inputs = next_observations
        inputs = observations

        target_actions = self._target_policy_f_prob_online(next_inputs)
        target_qvals = self._target_qf_f_prob_online(next_inputs,
                                                     target_actions)

        clip_range = (-self._clip_return,
                      0. if self._clip_pos_returns else self._clip_return)
        ys = np.clip(
            rewards + (1.0 - terminals) * self._discount * target_qvals,
            clip_range[0], clip_range[1])

        _, qval_loss, qval = self._f_train_qf(ys, inputs, actions)
        _, action_loss = self._f_train_policy(inputs)

        self._f_update_target()

        return qval_loss, ys, qval, action_loss
