"""
DQN from https://arxiv.org/pdf/1312.5602.pdf.

Known as Deep Q-Network, it estimates the Q-value function by deep neural
networks. It enables Q-Learning to be applied on high complexity environments.
To deal with pixel environments, numbers of tricks are usually needed, e.g.
skipping frames and stacking frames as single observation.
"""
import numpy as np
import tensorflow as tf

from garage.logger import tabular
from garage.misc.overrides import overrides
from garage.misc.tensor_utils import normalize_pixel_batch
from garage.tf.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.tf.misc import tensor_utils


class DQN(OffPolicyRLAlgorithm):
    """
    DQN.

    Args:
        env: Environment spec.
        max_path_length: Maximum path length. The episode will terminate when
            length of trajectory reaches max_path_length.
        qf_lr: Learning rate for Q-Function.
        qf_optimizer: Optimizer for Q-Function.
        discount: Discount factor for rewards.
        name: Name of the algorithm.
        target_network_update_freq: Frequency of updating target network.
        grad_norm_clipping: How much to clip gradient.
        double_q: Bool for using double q-network.
        print_freq: Log every print_freq steps.
    """

    def __init__(self,
                 env_spec,
                 replay_buffer,
                 max_path_length=None,
                 qf_lr=0.001,
                 qf_optimizer=tf.train.AdamOptimizer,
                 discount=1.0,
                 name='DQN',
                 target_network_update_freq=5,
                 grad_norm_clipping=None,
                 double_q=False,
                 **kwargs):
        self.qf_lr = qf_lr
        self.qf_optimizer = qf_optimizer
        self.name = name
        self.target_network_update_freq = target_network_update_freq
        self.grad_norm_clipping = grad_norm_clipping
        self.double_q = double_q

        self.episode_rewards = []
        self.episode_qf_losses = []

        super().__init__(
            env_spec=env_spec,
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
        action_dim = self.env_spec.action_space.n

        # build q networks
        with tf.name_scope(self.name, 'DQN'):
            self.action_t_ph = tf.placeholder(tf.int32, None, name='action')
            self.reward_t_ph = tf.placeholder(tf.float32, None, name='reward')
            self.done_t_ph = tf.placeholder(tf.float32, None, name='done')

            # clone a target q-function
            self.target_qf = self.qf.clone('target_qf')

            with tf.name_scope('update_ops'):
                target_update_op = tensor_utils.get_target_ops(
                    self.qf.get_global_vars(),
                    self.target_qf.get_global_vars())

            self._qf_update_ops = tensor_utils.compile_function(
                inputs=[], outputs=target_update_op)

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

                td_error = q_selected - tf.stop_gradient(target_q_values)
                loss = huber_loss(td_error)
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

            self._train_qf = tensor_utils.compile_function(
                inputs=[
                    self.qf.input, self.action_t_ph, self.reward_t_ph,
                    self.done_t_ph, self.target_qf.input
                ],
                outputs=[self._loss, self._optimize_loss])

    def train_once(self, itr, paths):
        """Train the algorithm once."""
        epoch = itr / self.n_epoch_cycles

        self.episode_rewards.extend(paths['undiscounted_returns'])
        last_average_return = np.mean(self.episode_rewards)
        for train_itr in range(self.n_train_steps):
            if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
                self.evaluate = True
                qf_loss = self.optimize_policy(epoch, None)
                self.episode_qf_losses.append(qf_loss)

        if self.evaluate:
            if itr % self.target_network_update_freq == 0:
                self._qf_update_ops()

        if itr % self.n_epoch_cycles == 0:
            if self.evaluate:
                mean100ep_rewards = round(
                    np.mean(self.episode_rewards[-100:]), 1)
                mean100ep_qf_loss = np.mean(self.episode_qf_losses[-100:])
                tabular.record('Epoch', epoch)
                tabular.record('AverageReturn', np.mean(self.episode_rewards))
                tabular.record('StdReturn', np.std(self.episode_rewards))
                tabular.record('Episode100RewardMean', mean100ep_rewards)
                tabular.record('{}/Episode100LossMean'.format(self.qf.name),
                               mean100ep_qf_loss)
        return last_average_return

    @overrides
    def get_itr_snapshot(self, itr):
        """Get snapshot of the policy."""
        return dict(itr=itr, policy=self.policy)

    @overrides
    def optimize_policy(self, itr, sample_data):
        """Optimize network using experiences from replay buffer."""
        transitions = self.replay_buffer.sample(self.buffer_batch_size)

        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        dones = transitions['terminal']

        observations = normalize_pixel_batch(self.env_spec, observations)
        next_observations = normalize_pixel_batch(self.env_spec,
                                                  next_observations)
        loss, _ = self._train_qf(observations, actions, rewards, dones,
                                 next_observations)

        return loss


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss."""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5, delta * (tf.abs(x) - 0.5 * delta))
