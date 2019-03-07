"""
This module implements a DDPG model.

DDPG, also known as Deep Deterministic Policy Gradient, uses actor-critic
method to optimize the policy and reward prediction. It uses a supervised
method to update the critic network and policy gradient to update the actor
network. And there are exploration strategy, replay buffer and target networks
involved to stabilize the training process.
"""
from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

import garage.misc.logger as logger
from garage.misc.overrides import overrides
from garage.tf.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.tf.misc import tensor_utils


class DDPG(OffPolicyRLAlgorithm):
    """
    A DDPG model based on https://arxiv.org/pdf/1509.02971.pdf.

    Example:
        $ python garage/examples/tf/ddpg_pendulum.py
    """

    def __init__(self,
                 env_spec,
                 replay_buffer,
                 target_update_tau=0.01,
                 policy_lr=1e-4,
                 qf_lr=1e-3,
                 policy_weight_decay=0,
                 qf_weight_decay=0,
                 policy_optimizer=tf.train.AdamOptimizer,
                 qf_optimizer=tf.train.AdamOptimizer,
                 clip_pos_returns=False,
                 clip_return=np.inf,
                 discount=0.99,
                 max_action=None,
                 name=None,
                 **kwargs):
        """
        Construct class.

        Args:
            env_spec(): Environment specification.
            target_update_tau(float): Interpolation parameter for doing the
        soft target update.
            discount(float): Discount factor for the cumulative return.
            policy_lr(float): Learning rate for training policy network.
            qf_lr(float): Learning rate for training q value network.
            policy_weight_decay(float): L2 weight decay factor for parameters
        of the policy network.
            qf_weight_decay(float): L2 weight decay factor for parameters
        of the q value network.
            policy_optimizer(): Optimizer for training policy network.
            qf_optimizer(): Optimizer for training q function network.
            clip_pos_returns(boolean): Whether or not clip positive returns.
            clip_return(float): Clip return to be in [-clip_return,
        clip_return].
            max_action(float): Maximum action magnitude.
            name(str): Name of the algorithm shown in computation graph.
        """
        action_bound = env_spec.action_space.high
        self.max_action = action_bound if max_action is None else max_action
        self.tau = target_update_tau
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.policy_weight_decay = policy_weight_decay
        self.qf_weight_decay = qf_weight_decay
        self.policy_optimizer = policy_optimizer
        self.qf_optimizer = qf_optimizer
        self.name = name
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.success_history = deque(maxlen=100)

        self.episode_rewards = []
        self.episode_policy_losses = []
        self.episode_qf_losses = []
        self.epoch_ys = []
        self.epoch_qs = []

        super(DDPG, self).__init__(
            env_spec=env_spec,
            replay_buffer=replay_buffer,
            use_target=True,
            discount=discount,
            **kwargs)

    @overrides
    def init_opt(self):
        with tf.name_scope(self.name, "DDPG"):
            # Create target policy and qf network
            self.target_policy_f_prob_online, _, _ = self.policy.build_net(
                trainable=False, name="target_policy")
            self.target_qf_f_prob_online, _, _, _ = self.qf.build_net(
                trainable=False, name="target_qf")

            # Set up target init and update function
            with tf.name_scope("setup_target"):
                policy_init_ops, policy_update_ops = get_target_ops(
                    self.policy.get_global_vars(),
                    self.policy.get_global_vars("target_policy"), self.tau)
                qf_init_ops, qf_update_ops = get_target_ops(
                    self.qf.get_global_vars(),
                    self.qf.get_global_vars("target_qf"), self.tau)
                target_init_op = policy_init_ops + qf_init_ops
                target_update_op = policy_update_ops + qf_update_ops

            f_init_target = tensor_utils.compile_function(
                inputs=[], outputs=target_init_op)
            f_update_target = tensor_utils.compile_function(
                inputs=[], outputs=target_update_op)

            with tf.name_scope("inputs"):
                if self.input_include_goal:
                    obs_dim = self.env_spec.observation_space.\
                        flat_dim_with_keys(["observation", "desired_goal"])
                else:
                    obs_dim = self.env_spec.observation_space.flat_dim
                y = tf.placeholder(tf.float32, shape=(None, 1), name="input_y")
                obs = tf.placeholder(
                    tf.float32,
                    shape=(None, obs_dim),
                    name="input_observation")
                actions = tf.placeholder(
                    tf.float32,
                    shape=(None, self.env_spec.action_space.flat_dim),
                    name="input_action")

            # Set up policy training function
            next_action = self.policy.get_action_sym(obs, name="policy_action")
            next_qval = self.qf.get_qval_sym(
                obs, next_action, name="policy_action_qval")
            with tf.name_scope("action_loss"):
                action_loss = -tf.reduce_mean(next_qval)
                if self.policy_weight_decay > 0.:
                    policy_reg = tc.layers.apply_regularization(
                        tc.layers.l2_regularizer(self.policy_weight_decay),
                        weights_list=self.policy.get_regularizable_vars())
                    action_loss += policy_reg

            with tf.name_scope("minimize_action_loss"):
                policy_train_op = self.policy_optimizer(
                    self.policy_lr, name="PolicyOptimizer").minimize(
                        action_loss, var_list=self.policy.get_trainable_vars())

            f_train_policy = tensor_utils.compile_function(
                inputs=[obs], outputs=[policy_train_op, action_loss])

            # Set up qf training function
            qval = self.qf.get_qval_sym(obs, actions, name="q_value")
            with tf.name_scope("qval_loss"):
                qval_loss = tf.reduce_mean(tf.squared_difference(y, qval))
                if self.qf_weight_decay > 0.:
                    qf_reg = tc.layers.apply_regularization(
                        tc.layers.l2_regularizer(self.qf_weight_decay),
                        weights_list=self.qf.get_regularizable_vars())
                    qval_loss += qf_reg

            with tf.name_scope("minimize_qf_loss"):
                qf_train_op = self.qf_optimizer(
                    self.qf_lr, name="QFunctionOptimizer").minimize(
                        qval_loss, var_list=self.qf.get_trainable_vars())

            f_train_qf = tensor_utils.compile_function(
                inputs=[y, obs, actions],
                outputs=[qf_train_op, qval_loss, qval])

            self.f_train_policy = f_train_policy
            self.f_train_qf = f_train_qf
            self.f_init_target = f_init_target
            self.f_update_target = f_update_target

    def train_once(self, itr, paths):
        epoch = itr / self.n_epoch_cycles

        self.episode_rewards.extend(paths["undiscounted_returns"])
        self.success_history.extend(paths["success_history"])
        last_average_return = np.mean(self.episode_rewards)
        self.log_diagnostics(paths)
        for train_itr in range(self.n_train_steps):
            if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
                self.evaluate = True
                qf_loss, y, q, policy_loss = self.optimize_policy(epoch, paths)

                self.episode_policy_losses.append(policy_loss)
                self.episode_qf_losses.append(qf_loss)
                self.epoch_ys.append(y)
                self.epoch_qs.append(q)

        if itr % self.n_epoch_cycles == 0:
            logger.log("Training finished")
            logger.log("Saving snapshot #{}".format(int(epoch)))
            params = self.get_itr_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            logger.log("Saved")
            if self.evaluate:
                logger.record_tabular('Epoch', epoch)
                logger.record_tabular('AverageReturn',
                                      np.mean(self.episode_rewards))
                logger.record_tabular('StdReturn',
                                      np.std(self.episode_rewards))
                logger.record_tabular('Policy/AveragePolicyLoss',
                                      np.mean(self.episode_policy_losses))
                logger.record_tabular('QFunction/AverageQFunctionLoss',
                                      np.mean(self.episode_qf_losses))
                logger.record_tabular('QFunction/AverageQ',
                                      np.mean(self.epoch_qs))
                logger.record_tabular('QFunction/MaxQ', np.max(self.epoch_qs))
                logger.record_tabular('QFunction/AverageAbsQ',
                                      np.mean(np.abs(self.epoch_qs)))
                logger.record_tabular('QFunction/AverageY',
                                      np.mean(self.epoch_ys))
                logger.record_tabular('QFunction/MaxY', np.max(self.epoch_ys))
                logger.record_tabular('QFunction/AverageAbsY',
                                      np.mean(np.abs(self.epoch_ys)))
                if self.input_include_goal:
                    logger.record_tabular('AverageSuccessRate',
                                          np.mean(self.success_history))

            if not self.smooth_return:
                self.episode_rewards = []
                self.episode_policy_losses = []
                self.episode_qf_losses = []
                self.epoch_ys = []
                self.epoch_qs = []

            self.success_history.clear()

        return last_average_return

    @overrides
    def optimize_policy(self, itr, samples_data):
        """
        Perform algorithm optimizing.

        Returns:
            action_loss: Loss of action predicted by the policy network.
            qval_loss: Loss of q value predicted by the q network.
            ys: y_s.
            qval: Q value predicted by the q network.

        """
        transitions = self.replay_buffer.sample(self.buffer_batch_size)
        observations = transitions["observation"]
        rewards = transitions["reward"]
        actions = transitions["action"]
        next_observations = transitions["next_observation"]
        terminals = transitions["terminal"]

        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        if self.input_include_goal:
            goals = transitions["goal"]
            next_inputs = np.concatenate((next_observations, goals), axis=-1)
            inputs = np.concatenate((observations, goals), axis=-1)
        else:
            next_inputs = next_observations
            inputs = observations

        target_actions = self.target_policy_f_prob_online(next_inputs)
        target_qvals = self.target_qf_f_prob_online(next_inputs,
                                                    target_actions)

        clip_range = (-self.clip_return,
                      0. if self.clip_pos_returns else self.clip_return)
        ys = np.clip(
            rewards + (1.0 - terminals) * self.discount * target_qvals,
            clip_range[0], clip_range[1])

        _, qval_loss, qval = self.f_train_qf(ys, inputs, actions)
        _, action_loss = self.f_train_policy(inputs)

        self.f_update_target()

        return qval_loss, ys, qval, action_loss

    @overrides
    def get_itr_snapshot(self, itr):
        return dict(itr=itr, policy=self.policy)


def get_target_ops(variables, target_variables, tau):
    """Get target network update operations."""
    update_ops = []
    init_ops = []
    assert len(variables) == len(target_variables)
    for var, target_var in zip(variables, target_variables):
        init_ops.append(tf.assign(target_var, var))
        update_ops.append(
            tf.assign(target_var, tau * var + (1.0 - tau) * target_var))
    return init_ops, update_ops
