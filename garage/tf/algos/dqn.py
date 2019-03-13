"""
DQN from https://arxiv.org/pdf/1312.5602.pdf.

Known as Deep Q-Network, it estimates the Q-value function by deep neural
networks. It enables Q-Learning to be applied on high complexity environments.
To deal with pixel environments, numbers of tricks are usually needed, e.g.
skipping frames and stacking frames as single observation.
"""
import numpy as np
import tensorflow as tf

from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.tf.misc import tensor_utils


class DQN(OffPolicyRLAlgorithm):
    """
    DQN.

    Args:
        env: Environment spec.
        max_path_length: Maximum path length. The episode will terminate when
            length of trajectory reaches max_path_length.
        num_timesteps: Total number of timesteps.
        qf_lr: Learning rate for Q-Function.
        qf_optimizer: Optimizer for Q-Function.
        discount: Discount factor for rewards.
        name: Name of the algorithm.
        target_network_update_freq: Frequency of updating target network.
        grad_norm_clipping: How much to clip gradient.
        double_q: Bool for using double q-network.
        print_freq: Log every print_freq steps.
        use_atari_wrappers: Bool for using atari wrappers. If True,
            observations need to be processed before use.
    """

    def __init__(self,
                 env,
                 replay_buffer,
                 max_path_length=200,
                 num_timesteps=int(1e5),
                 qf_lr=0.001,
                 qf_optimizer=tf.train.AdamOptimizer,
                 discount=1.0,
                 name='DQN',
                 target_network_update_freq=5,
                 grad_norm_clipping=None,
                 double_q=False,
                 print_freq=1000,
                 use_atari_wrappers=False,
                 **kwargs):
        self.qf_lr = qf_lr
        self.qf_optimizer = qf_optimizer
        self.name = name
        self.target_network_update_freq = target_network_update_freq
        self.grad_norm_clipping = grad_norm_clipping
        self.double_q = double_q
        self.print_freq = print_freq
        self.num_timesteps = num_timesteps
        self.use_atari_wrappers = use_atari_wrappers

        super().__init__(
            env=env,
            replay_buffer=replay_buffer,
            max_path_length=max_path_length,
            discount=discount,
            **kwargs)

    @overrides
    def init_opt(self):
        """
        Initialize the networks and Ops.

        Assume discrete space for dqn, so action dimension
        will always be action_space.n
        """
        action_dim = self.env.action_space.n

        # build q networks
        with tf.name_scope(self.name, 'DQN'):
            self.action_t_ph = tf.placeholder(tf.int32, None, name="action")
            self.reward_t_ph = tf.placeholder(tf.float32, None, name="reward")
            self.done_t_ph = tf.placeholder(tf.float32, None, name="done")

            # clone a target q-function
            self.target_qf = self.qf.clone('target_qf')

            with tf.name_scope('update_ops'):
                self._qf_update_ops = tensor_utils.get_target_ops(
                    self.qf.get_global_vars(),
                    self.target_qf.get_global_vars())

            with tf.name_scope('td_error'):
                # Q-value of the selected action
                action = tf.one_hot(self.action_t_ph, action_dim)
                q_selected = tf.reduce_sum(
                    self.qf.q_vals * action,  # yapf: disable
                    axis=1)

                # r + Q'(s', argmax_a(Q(s', _)) - Q(s, a)
                if self.double_q:
                    target_qval_with_online_q = self.qf.get_qval_sym(
                        self.target_qf.input, self.qf.name)
                    future_best_q_val_action = tf.argmax(
                        target_qval_with_online_q, 1)
                    future_best_q_val = tf.reduce_sum(
                        self.target_qf.q_vals * tf.one_hot(
                            future_best_q_val_action, action_dim),
                        axis=1)
                else:
                    # r + max_a(Q'(s', _)) - Q(s, a)
                    future_best_q_val = tf.reduce_max(
                        self.target_qf.q_vals, axis=1)

                q_best_masked = (1.0 - self.done_t_ph) * future_best_q_val
                # if done, it's just reward
                # else reward + discount * future_best_q_val
                target_q_values = self.reward_t_ph + self.discount * q_best_masked  # noqa: E501

                td_error = tf.stop_gradient(target_q_values) - q_selected
                loss = tf.square(td_error)
                self._loss = tf.reduce_mean(loss)

            with tf.name_scope('optimize_ops'):
                optimizer = self.qf_optimizer(self.qf_lr)
                if self.grad_norm_clipping is not None:
                    gradients = optimizer.compute_gradients(
                        self._loss, var_list=self.qf.get_trainable_vars())
                    for i, (grad, var) in enumerate(gradients):
                        if grad is not None:
                            gradients[i] = (tf.clip_by_norm(
                                grad, self.grad_norm_clipping), var)
                    self._optimize_loss = optimizer.apply_gradients(gradients)
                else:
                    self._optimize_loss = optimizer.minimize(
                        self._loss, var_list=self.qf.get_trainable_vars())

    @overrides
    def train(self, sess=None):
        """
        Train the network.

        A tf.Session can be provided, or will be created otherwise.
        """
        created_session = True if sess is None else False
        if sess is None:
            self.sess = tf.Session()
            self.sess.__enter__()
        else:
            self.sess = sess

        self.sess.run(tf.global_variables_initializer())

        episode_rewards = []
        episode_qf_losses = []
        episode_length = []
        ts = 0

        self.sess.run(self._qf_update_ops, feed_dict=dict())

        obs = self.env.reset()

        # Baselines atari wrappers use a LazyFrame object to store the
        # observation. The way to unpack it is to unpack obs as numpy array.
        # For reference, see
        # https://github.com/openai/baselines/blob/1259f6ab25c6f7261e33c
        # 4c3b92df869188f9260/baselines/common/atari_wrappers.py#L202
        if self.use_atari_wrappers:
            obs = np.asarray(obs)
        episode_rewards.append(0.)

        for itr in range(self.num_timesteps):
            with logger.prefix('Timestep #%d | ' % itr):
                obs_normalized = tensor_utils.normalize_pixel_batch(
                    self.env.spec, obs)

                if self.es:
                    action, _ = self.es.get_action(itr, obs_normalized,
                                                   self.policy)
                else:
                    action, _ = self.policy.get_action(obs_normalized)

                next_obs, reward, done, env_info = self.env.step(action)

                if self.use_atari_wrappers:
                    next_obs = np.asarray(next_obs)

                self.replay_buffer.add_transition(
                    observation=obs,
                    action=action,
                    reward=reward,
                    terminal=done,
                    next_observation=next_obs,
                )

                episode_rewards[-1] += reward
                ts += 1
                obs = next_obs

                if done:
                    episode_length.append(ts)
                    ts = 0
                    episode_rewards.append(0.)
                    obs = self.env.reset()
                    if self.use_atari_wrappers:
                        obs = np.asarray(obs)

                if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
                    self.evaluate = True
                    qf_loss = self.optimize_policy(itr, None)
                    episode_qf_losses.append(qf_loss)
                    if itr % self.target_network_update_freq == 0:
                        self.sess.run(self._qf_update_ops, feed_dict=dict())

                if self.plot:
                    self.plotter.update_plot(self.policy, self.max_path_length)
                    if self.pause_for_plot:
                        input('Plotting evaluation run: Press Enter to '
                              'continue...')

                if self.evaluate:
                    mean100ep_rewards = round(
                        np.mean(episode_rewards[-100:]), 1)
                    mean100ep_qf_loss = np.mean(episode_qf_losses[-100:])
                    if itr % self.print_freq == 0:
                        logger.record_tabular('Iteration', itr)
                        logger.record_tabular('Episode100RewardMean',
                                              mean100ep_rewards)
                        logger.record_tabular('StdReturn',
                                              np.std(episode_rewards))
                        logger.record_tabular('{}/Episode100LossMean'.format(self.qf.name),
                                              mean100ep_qf_loss)
                        logger.record_tabular('AverageEpisodeLength',
                                              np.mean(episode_length))

                logger.dump_tabular(with_prefix=False)

        if created_session:
            self.sess.close()

        return mean100ep_rewards

    @overrides
    def optimize_policy(self, itr, sample_data):
        """Optimize network using experiences from replay buffer."""
        transitions = self.replay_buffer.sample(self.buffer_batch_size)

        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        dones = transitions['terminal']

        observations = tensor_utils.normalize_pixel_batch(self.env.spec, observations)
        next_observations = tensor_utils.normalize_pixel_batch(
            self.env.spec, next_observations)

        loss, _ = self.sess.run(
            [self._loss, self._optimize_loss],
            feed_dict={
                self.qf.input: observations,
                self.action_t_ph: actions,
                self.reward_t_ph: rewards,
                self.done_t_ph: dones,
                self.target_qf.input: next_observations
            })

        return loss