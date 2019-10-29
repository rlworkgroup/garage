from dowel import tabular
import numpy as np
import tensorflow as tf

from garage.misc.tensor_utils import normalize_pixel_batch
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
        policy (garage.tf.policies.base.Policy): Policy.
        qf (object): The q value network.
        replay_buffer (garage.replay_buffer.ReplayBuffer): Replay buffer.
        exploration_strategy (garage.np.exploration_strategies.
            ExplorationStrategy): Exploration strategy.
        n_epoch_cycles (int): Epoch cycles.
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
        input_include_goal (bool): Whether input includes goal.
        smooth_return (bool): Whether to smooth the return.
        name (str): Name of the algorithm.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 qf,
                 replay_buffer,
                 exploration_strategy=None,
                 n_epoch_cycles=20,
                 min_buffer_size=int(1e4),
                 buffer_batch_size=64,
                 rollout_batch_size=1,
                 n_train_steps=50,
                 max_path_length=None,
                 qf_lr=0.001,
                 qf_optimizer=tf.compat.v1.train.AdamOptimizer,
                 discount=1.0,
                 target_network_update_freq=5,
                 grad_norm_clipping=None,
                 double_q=False,
                 reward_scale=1.,
                 input_include_goal=False,
                 smooth_return=True,
                 name='DQN'):
        self.qf_lr = qf_lr
        self.qf_optimizer = qf_optimizer
        self.name = name
        self.target_network_update_freq = target_network_update_freq
        self.grad_norm_clipping = grad_norm_clipping
        self.double_q = double_q

        # clone a target q-function
        self.target_qf = qf.clone('target_qf')

        super(DQN, self).__init__(env_spec=env_spec,
                                  policy=policy,
                                  qf=qf,
                                  exploration_strategy=exploration_strategy,
                                  min_buffer_size=min_buffer_size,
                                  n_train_steps=n_train_steps,
                                  n_epoch_cycles=n_epoch_cycles,
                                  buffer_batch_size=buffer_batch_size,
                                  rollout_batch_size=rollout_batch_size,
                                  replay_buffer=replay_buffer,
                                  max_path_length=max_path_length,
                                  discount=discount,
                                  reward_scale=reward_scale,
                                  input_include_goal=input_include_goal,
                                  smooth_return=smooth_return)

    def init_opt(self):
        """
        Initialize the networks and Ops.

        Assume discrete space for dqn, so action dimension
        will always be action_space.n
        """
        action_dim = self.env_spec.action_space.n

        self.episode_rewards = []
        self.episode_qf_losses = []

        # build q networks
        with tf.name_scope(self.name, 'DQN'):
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
                    self.target_qf.get_global_vars())

            self._qf_update_ops = tensor_utils.compile_function(
                inputs=[], outputs=target_update_op)

            with tf.name_scope('td_error'):
                # Q-value of the selected action
                action = tf.one_hot(action_t_ph, action_dim)
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
                        self.target_qf.q_vals *
                        tf.one_hot(future_best_q_val_action, action_dim),
                        axis=1)
                else:
                    # r + max_a(Q'(s', _)) - Q(s, a)
                    future_best_q_val = tf.reduce_max(self.target_qf.q_vals,
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
                optimizer = self.qf_optimizer(self.qf_lr)
                if self.grad_norm_clipping is not None:
                    gradients = optimizer.compute_gradients(
                        loss, var_list=self.qf.get_trainable_vars())
                    for i, (grad, var) in enumerate(gradients):
                        if grad is not None:
                            gradients[i] = (tf.clip_by_norm(
                                grad, self.grad_norm_clipping), var)
                        optimize_loss = optimizer.apply_gradients(gradients)
                else:
                    optimize_loss = optimizer.minimize(
                        loss, var_list=self.qf.get_trainable_vars())

            self._train_qf = tensor_utils.compile_function(
                inputs=[
                    self.qf.input, action_t_ph, reward_t_ph, done_t_ph,
                    self.target_qf.input
                ],
                outputs=[loss, optimize_loss])

    def train_once(self, itr, paths):
        """Train the algorithm once."""
        paths = self.process_samples(itr, paths)
        epoch = itr / self.n_epoch_cycles

        self.episode_rewards.extend(paths['undiscounted_returns'])
        last_average_return = np.mean(self.episode_rewards)
        for train_itr in range(self.n_train_steps):
            if (self.replay_buffer.n_transitions_stored >=
                    self.min_buffer_size):
                self.evaluate = True
                qf_loss = self.optimize_policy(epoch, None)
                self.episode_qf_losses.append(qf_loss)

        if self.evaluate:
            if itr % self.target_network_update_freq == 0:
                self._qf_update_ops()

        if itr % self.n_epoch_cycles == 0:
            if self.evaluate:
                mean100ep_rewards = round(np.mean(self.episode_rewards[-100:]),
                                          1)
                mean100ep_qf_loss = np.mean(self.episode_qf_losses[-100:])
                tabular.record('Epoch', epoch)
                tabular.record('AverageReturn', np.mean(self.episode_rewards))
                tabular.record('StdReturn', np.std(self.episode_rewards))
                tabular.record('Episode100RewardMean', mean100ep_rewards)
                tabular.record('{}/Episode100LossMean'.format(self.qf.name),
                               mean100ep_qf_loss)
        return last_average_return

    def get_itr_snapshot(self, itr):
        """Get snapshot of the policy."""
        return dict(itr=itr, policy=self.policy)

    def optimize_policy(self, itr, sample_data):
        """Optimize network using experiences from replay buffer."""
        transitions = self.replay_buffer.sample(self.buffer_batch_size)

        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        dones = transitions['terminal']

        # normalize pixel to range [0, 1] since the samples stored in the
        # replay buffer are of type uint8 and not normalized, for memory
        # optimization
        observations = normalize_pixel_batch(self.env_spec, observations)
        next_observations = normalize_pixel_batch(self.env_spec,
                                                  next_observations)
        loss, _ = self._train_qf(observations, actions, rewards, dones,
                                 next_observations)

        return loss

    def __getstate__(self):
        data = self.__dict__.copy()
        del data['_qf_update_ops']
        del data['_train_qf']
        return data

    def __setstate__(self, state):
        self.__dict__ = state
        self.init_opt()
