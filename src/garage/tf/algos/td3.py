"""This module implements a TD3 model.

TD3, or Twin Delayed Deep Deterministic Policy Gradient, uses actor-critic
method to optimize the policy and reward prediction. Notably, it uses the
minimum value of two critics instead of one to limit overestimation.
"""

from dowel import logger, tabular
import numpy as np
import tensorflow as tf

from garage import _Default, log_performance, make_optimizer
from garage.np import obtain_evaluation_episodes
from garage.np.algos import RLAlgorithm
from garage.sampler import FragmentWorker, LocalSampler
from garage.tf.misc import tensor_utils


class TD3(RLAlgorithm):
    """Implementation of TD3.

    Based on https://arxiv.org/pdf/1802.09477.pdf.

    Example:
        $ python garage/examples/tf/td3_pendulum.py

    Args:
        env_spec (EnvSpec): Environment.
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
        max_episode_length (int): Maximum length of an episode.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If `None`, defaults to
            `max_episode_length`.
        n_train_steps (int): Number of optimizations in each epoch cycle.
        buffer_batch_size (int): Size of replay buffer.
        min_buffer_size (int):
            Number of samples in replay buffer before first optimization.
        reward_scale (float): Scale to reward.
        exploration_policy_sigma (float): Action noise sigma.
        exploration_policy_clip (float): Action noise clip.
        actor_update_period (int): Action update period.
        exploration_policy (ExplorationPolicy): Exploration strategy.

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
            max_episode_length=None,
            max_episode_length_eval=None,
            n_train_steps=50,
            buffer_batch_size=64,
            min_buffer_size=1e4,
            reward_scale=1.,
            exploration_policy_sigma=0.2,
            actor_update_period=2,
            exploration_policy_clip=0.5,
            exploration_policy=None):
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

        self.qf2 = qf2
        self.qf = qf

        self._exploration_policy_sigma = exploration_policy_sigma
        self._exploration_policy_clip = exploration_policy_clip
        self._actor_update_period = actor_update_period
        self._action_loss = None

        self._target_qf2 = qf2.clone('target_qf2')
        self._policy_optimizer = policy_optimizer
        self._qf_optimizer = qf_optimizer
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr

        self._policy = policy
        self._n_train_steps = n_train_steps

        self._min_buffer_size = min_buffer_size
        self._qf = qf
        self._steps_per_epoch = steps_per_epoch
        self._n_train_steps = n_train_steps
        self._buffer_batch_size = buffer_batch_size
        self._discount = discount
        self._reward_scale = reward_scale
        self.max_episode_length = max_episode_length
        self._max_episode_length_eval = max_episode_length_eval
        self._eval_env = None

        self.env_spec = env_spec
        self.replay_buffer = replay_buffer
        self.policy = policy
        self.exploration_policy = exploration_policy

        self.sampler_cls = LocalSampler
        self.worker_cls = FragmentWorker

        self.init_opt()

    def init_opt(self):
        """Build the loss function and init the optimizer."""
        with tf.name_scope(self._name):
            # Create target policy (actor) and qf (critic) networks
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

            policy_network_outputs = self._target_policy.build(obs,
                                                               name='policy')
            target_qf_outputs = self._target_qf.build(obs, actions, name='qf')
            target_qf2_outputs = self._target_qf2.build(obs,
                                                        actions,
                                                        name='qf')

            self.target_policy_f_prob_online = tensor_utils.compile_function(
                inputs=[obs], outputs=policy_network_outputs)

            self.target_qf_f_prob_online = tensor_utils.compile_function(
                inputs=[obs, actions], outputs=target_qf_outputs)

            self.target_qf2_f_prob_online = tensor_utils.compile_function(
                inputs=[obs, actions], outputs=target_qf2_outputs)

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

            # Set up policy training function
            next_action = self.policy.build(obs, name='policy_action')
            next_qval = self.qf.build(obs,
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
            qval = self.qf.build(obs, actions, name='q_value')
            q2val = self.qf2.build(obs, actions, name='q2_value')
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

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): Experiment runner, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = runner.get_env_copy()
        last_returns = [float('nan')]
        runner.enable_logging = False

        for _ in runner.step_epochs():
            for cycle in range(self._steps_per_epoch):
                runner.step_path = runner.obtain_episodes(runner.step_itr)
                self.train_once(runner.step_itr, runner.step_path)
                if (cycle == 0 and self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    runner.enable_logging = True
                    eval_episodes = obtain_evaluation_episodes(
                        self.policy, self._eval_env)
                    last_returns = log_performance(runner.step_itr,
                                                   eval_episodes,
                                                   discount=self._discount)
                runner.step_itr += 1

        return np.mean(last_returns)

    def train_once(self, itr, episodes):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.

        """
        self.replay_buffer.add_episode_batch(episodes)

        epoch = itr / self._steps_per_epoch

        for _ in range(self._n_train_steps):
            if (self.replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                qf_loss, y_s, qval, policy_loss = self.optimize_policy(itr)

                self._episode_policy_losses.append(policy_loss)
                self._episode_qf_losses.append(qf_loss)
                self._epoch_ys.append(y_s)
                self._epoch_qs.append(qval)

        if itr % self._steps_per_epoch == 0:
            logger.log('Training finished')

            if (self.replay_buffer.n_transitions_stored >=
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

    def optimize_policy(self, itr):
        """Perform algorithm optimizing.

        Args:
            itr(int): Iterations.

        Returns:
            float: Loss of action predicted by the policy network.
            float: Loss of q value predicted by the q network.
            float: y_s.
            float: Q value predicted by the q network.

        """
        transitions = self.replay_buffer.sample_transitions(
            self._buffer_batch_size)

        observations = transitions['observations']
        rewards = transitions['rewards']
        actions = transitions['actions']
        next_observations = transitions['next_observations']
        terminals = transitions['terminals']

        rewards *= self._reward_scale

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
        ys = (rewards + (1.0 - terminals) * self._discount * target_qvals)

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
