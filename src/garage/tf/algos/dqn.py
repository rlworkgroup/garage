"""Deep Q-Learning Network algorithm."""
import akro
from dowel import tabular
import numpy as np
import tensorflow as tf

from garage import _Default, make_optimizer
from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.tf.misc import tensor_utils


class DQN(OffPolicyRLAlgorithm):
    """DQN from https://arxiv.org/pdf/1312.5602.pdf.

    Known as Deep Q-Network, it estimates the Q-value function by deep neural
    networks. It enables Q-Learning to be applied on high complexity
    environments. To deal with pixel environments, numbers of tricks are
    usually needed, e.g. skipping frames and stacking frames as single
    observation.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        policy (garage.tf.policies.Policy): Policy.
        qf (object): The q value network.
        replay_buffer (garage.replay_buffer.ReplayBuffer): Replay buffer.
        exploration_policy
            (garage.np.exploration_policies.ExplorationPolicy):
            Exploration strategy.
        steps_per_epoch (int): Number of train_once calls per epoch.
        min_buffer_size (int): The minimum buffer size for replay buffer.
        buffer_batch_size (int): Batch size for replay buffer.
        rollout_batch_size (int): Roll out batch size.
        n_train_steps (int): Training steps.
        max_path_length (int): Maximum path length. The episode will
            terminate when length of trajectory reaches max_path_length.
        qf_lr (float): Learning rate for Q-Function.
        qf_optimizer (tf.Optimizer): Optimizer for Q-Function.
        discount (float): Discount factor for rewards.
        target_network_update_freq (int): Frequency of updating target
            network.
        grad_norm_clipping (float): Maximum clipping value for clipping
            tensor values to a maximum L2-norm. It must be larger than 0.
            If None, no gradient clipping is done. For detail, see
            docstring for tf.clip_by_norm.
        double_q (bool): Bool for using double q-network.
        reward_scale (float): Reward scale.
        smooth_return (bool): Whether to smooth the return.
        name (str): Name of the algorithm.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 qf,
                 replay_buffer,
                 exploration_policy=None,
                 steps_per_epoch=20,
                 min_buffer_size=int(1e4),
                 buffer_batch_size=64,
                 rollout_batch_size=1,
                 n_train_steps=50,
                 max_path_length=None,
                 qf_lr=_Default(0.001),
                 qf_optimizer=tf.compat.v1.train.AdamOptimizer,
                 discount=1.0,
                 target_network_update_freq=5,
                 grad_norm_clipping=None,
                 double_q=False,
                 reward_scale=1.,
                 smooth_return=True,
                 name='DQN'):
        self._qf_optimizer = qf_optimizer
        self._qf_lr = qf_lr
        self._name = name
        self._target_network_update_freq = target_network_update_freq
        self._grad_norm_clipping = grad_norm_clipping
        self._double_q = double_q

        # clone a target q-function
        self._target_qf = qf.clone('target_qf')

        super(DQN, self).__init__(env_spec=env_spec,
                                  policy=policy,
                                  qf=qf,
                                  exploration_policy=exploration_policy,
                                  min_buffer_size=min_buffer_size,
                                  n_train_steps=n_train_steps,
                                  steps_per_epoch=steps_per_epoch,
                                  buffer_batch_size=buffer_batch_size,
                                  rollout_batch_size=rollout_batch_size,
                                  replay_buffer=replay_buffer,
                                  max_path_length=max_path_length,
                                  discount=discount,
                                  reward_scale=reward_scale,
                                  smooth_return=smooth_return)

    def init_opt(self):
        """Initialize the networks and Ops.

        Assume discrete space for dqn, so action dimension
        will always be action_space.n
        """
        action_dim = self.env_spec.action_space.n

        self.episode_rewards = []
        self.episode_qf_losses = []

        # build q networks
        with tf.name_scope(self._name):
            action_t_ph = tf.compat.v1.placeholder(tf.int32,
                                                   None,
                                                   name='action')
            reward_t_ph = tf.compat.v1.placeholder(tf.float32,
                                                   None,
                                                   name='reward')
            done_t_ph = tf.compat.v1.placeholder(tf.float32, None, name='done')

            with tf.name_scope('update_ops'):
                target_update_op = tensor_utils.get_target_ops(
                    self.qf.get_global_vars(),
                    self._target_qf.get_global_vars())

            self._qf_update_ops = tensor_utils.compile_function(
                inputs=[], outputs=target_update_op)

            with tf.name_scope('td_error'):
                # Q-value of the selected action
                action = tf.one_hot(action_t_ph,
                                    action_dim,
                                    on_value=1.,
                                    off_value=0.)
                q_selected = tf.reduce_sum(
                    self.qf.q_vals * action,  # yapf: disable
                    axis=1)

                # r + Q'(s', argmax_a(Q(s', _)) - Q(s, a)
                if self._double_q:
                    target_qval_with_online_q = self.qf.get_qval_sym(
                        self._target_qf.input, self.qf.name)
                    future_best_q_val_action = tf.argmax(
                        target_qval_with_online_q, 1)
                    future_best_q_val = tf.reduce_sum(
                        self._target_qf.q_vals *
                        tf.one_hot(future_best_q_val_action,
                                   action_dim,
                                   on_value=1.,
                                   off_value=0.),
                        axis=1)
                else:
                    # r + max_a(Q'(s', _)) - Q(s, a)
                    future_best_q_val = tf.reduce_max(self._target_qf.q_vals,
                                                      axis=1)

                q_best_masked = (1.0 - done_t_ph) * future_best_q_val
                # if done, it's just reward
                # else reward + discount * future_best_q_val
                target_q_values = (reward_t_ph + self.discount * q_best_masked)

                # td_error = q_selected - tf.stop_gradient(target_q_values)
                loss = tf.compat.v1.losses.huber_loss(
                    q_selected, tf.stop_gradient(target_q_values))
                loss = tf.reduce_mean(loss)

            with tf.name_scope('optimize_ops'):
                qf_optimizer = make_optimizer(self._qf_optimizer,
                                              learning_rate=self._qf_lr)
                if self._grad_norm_clipping is not None:
                    gradients = qf_optimizer.compute_gradients(
                        loss, var_list=self.qf.get_trainable_vars())
                    for i, (grad, var) in enumerate(gradients):
                        if grad is not None:
                            gradients[i] = (tf.clip_by_norm(
                                grad, self._grad_norm_clipping), var)
                        optimize_loss = qf_optimizer.apply_gradients(gradients)
                else:
                    optimize_loss = qf_optimizer.minimize(
                        loss, var_list=self.qf.get_trainable_vars())

            self._train_qf = tensor_utils.compile_function(
                inputs=[
                    self.qf.input, action_t_ph, reward_t_ph, done_t_ph,
                    self._target_qf.input
                ],
                outputs=[loss, optimize_loss])

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Average return.

        """
        paths = self.process_samples(itr, paths)
        epoch = itr / self.steps_per_epoch

        self.episode_rewards.extend(paths['undiscounted_returns'])
        last_average_return = np.mean(self.episode_rewards)
        for _ in range(self.n_train_steps):
            if self._buffer_prefilled:
                qf_loss = self.optimize_policy(None)
                self.episode_qf_losses.append(qf_loss)

        if self._buffer_prefilled:
            if itr % self._target_network_update_freq == 0:
                self._qf_update_ops()

        if itr % self.steps_per_epoch == 0:
            if self._buffer_prefilled:
                mean100ep_rewards = round(np.mean(self.episode_rewards[-100:]),
                                          1)
                mean100ep_qf_loss = np.mean(self.episode_qf_losses[-100:])
                tabular.record('Epoch', epoch)
                tabular.record('Episode100RewardMean', mean100ep_rewards)
                tabular.record('{}/Episode100LossMean'.format(self.qf.name),
                               mean100ep_qf_loss)
        return last_average_return

    def optimize_policy(self, samples_data):
        """Optimize network using experiences from replay buffer.

        Args:
            samples_data (list): Processed batch data.

        Returns:
            numpy.float64: Loss of policy.

        """
        del samples_data

        transitions = self.replay_buffer.sample_transitions(
            self.buffer_batch_size)

        observations = transitions['observations']
        rewards = transitions['rewards']
        actions = self.env_spec.action_space.unflatten_n(
            transitions['actions'])
        next_observations = transitions['next_observations']
        dones = transitions['terminals']

        if isinstance(self.env_spec.observation_space, akro.Image):
            if len(observations.shape[1:]) < len(
                    self.env_spec.observation_space.shape):
                observations = self.env_spec.observation_space.unflatten_n(
                    observations)
                next_observations = self.env_spec.observation_space.\
                    unflatten_n(next_observations)

        loss, _ = self._train_qf(observations, actions, rewards, dones,
                                 next_observations)

        return loss

    def __getstate__(self):
        """Parameters to save in snapshot.

        Returns:
            dict: Parameters to save.

        """
        data = self.__dict__.copy()
        del data['_qf_update_ops']
        del data['_train_qf']
        return data

    def __setstate__(self, state):
        """Parameters to restore from snapshot.

        Args:
            state (dict): Parameters to restore from.

        """
        self.__dict__ = state
        self.init_opt()
