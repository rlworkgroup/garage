"""This modules creates a DDPG model in PyTorch."""
from collections import deque
import copy

from dowel import logger, tabular
import numpy as np
import torch

from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.torch.utils import np_to_torch, torch_to_np


class DDPG(OffPolicyRLAlgorithm):
    """A DDPG model implemented with PyTorch.

    DDPG, also known as Deep Deterministic Policy Gradient, uses actor-critic
    method to optimize the policy and Q-function prediction. It uses a
    supervised method to update the critic network and policy gradient to
    update the actor network. And there are exploration strategy, replay
    buffer and target networks involved to stabilize the training process.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.base.Policy): Policy.
        target_policy (garage.torch.policies.base.Policy): Copy of policy.
        qf (object): Q-value network.
        target_qf (object): Copy of Q-value network.
        replay_buffer (garage.replay_buffer.ReplayBuffer): Replay buffer.
        n_epoch_cycles (int): Number of train_once calls per epoch.
        n_train_steps (int): Training steps.
        max_path_length (int): Maximum path length. The episode will
            terminate when length of trajectory reaches max_path_length.
        min_buffer_size (int): The minimum buffer size for replay buffer.
        rollout_batch_size (int): Roll out batch size.
        exploration_strategy (garage.np.exploration_strategies.
            ExplorationStrategy): Exploration strategy.
        target_update_tau (float): Interpolation parameter for doing the
            soft target update.
        policy_lr (float): Learning rate for training policy network.
        qf_lr (float): Learning rate for training Q-value network.
        discount(float): Discount factor for the cumulative return.
        policy_weight_decay (float): L2 weight decay factor for parameters
            of the policy network.
        qf_weight_decay (float): L2 weight decay factor for parameters
            of the q value network.
        optimizer (torch.optimizer): Optimizer for training policy network
            and Q-vaule network.
        criterion (torch.nn.modules.loss): Loss function for Q-value newtork.
        clip_pos_returns (bool): Whether or not clip positive returns.
        clip_return (float): Clip return to be in [-clip_return,
            clip_return].
        max_action (float): Maximum action magnitude.
        reward_scale (float): Reward scale.
        smooth_return (bool): Whether to smooth the return for logging.
        name (str): Name of the algorithm shown in computation graph.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 qf,
                 replay_buffer,
                 n_epoch_cycles=20,
                 n_train_steps=50,
                 max_path_length=None,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 rollout_batch_size=1,
                 exploration_strategy=None,
                 target_update_tau=0.01,
                 policy_lr=1e-4,
                 qf_lr=1e-3,
                 discount=0.99,
                 policy_weight_decay=0,
                 qf_weight_decay=0,
                 optimizer=torch.optim.Adam,
                 clip_pos_returns=False,
                 clip_return=np.inf,
                 max_action=None,
                 reward_scale=1.,
                 smooth_return=True):
        action_bound = env_spec.action_space.high
        self.tau = target_update_tau
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.policy_weight_decay = policy_weight_decay
        self.qf_weight_decay = qf_weight_decay
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.max_action = action_bound if max_action is None else max_action
        self.evaluate = False

        self.success_history = deque(maxlen=100)
        self.episode_rewards = []
        self.episode_policy_losses = []
        self.episode_qf_losses = []
        self.epoch_ys = []
        self.epoch_qs = []

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         qf=qf,
                         n_train_steps=n_train_steps,
                         n_epoch_cycles=n_epoch_cycles,
                         max_path_length=max_path_length,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         rollout_batch_size=rollout_batch_size,
                         exploration_strategy=exploration_strategy,
                         replay_buffer=replay_buffer,
                         use_target=True,
                         discount=discount,
                         reward_scale=reward_scale,
                         smooth_return=smooth_return)

        self.target_policy = copy.deepcopy(self.policy)
        self.target_qf = copy.deepcopy(self.qf)
        self.policy_optimizer = optimizer(self.policy.parameters(),
                                          lr=self.policy_lr)
        self.qf_optimizer = optimizer(self.qf.parameters(), lr=self.qf_lr)

    def train_once(self, itr, paths):
        """Perform one iteration of training."""
        paths = self.process_samples(itr, paths)

        epoch = itr / self.n_epoch_cycles

        self.episode_rewards.extend([
            path for path, complete in zip(paths['undiscounted_returns'],
                                           paths['complete']) if complete
        ])
        self.success_history.extend([
            path for path, complete in zip(paths['success_history'],
                                           paths['complete']) if complete
        ])

        last_average_return = np.mean(self.episode_rewards)
        for train_itr in range(self.n_train_steps):
            if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
                self.evaluate = True
                samples = self.replay_buffer.sample(self.buffer_batch_size)
                qf_loss, y, q, policy_loss = torch_to_np(
                    self.optimize_policy(itr, samples))

                self.episode_policy_losses.append(policy_loss)
                self.episode_qf_losses.append(qf_loss)
                self.epoch_ys.append(y)
                self.epoch_qs.append(q)

        if itr % self.n_epoch_cycles == 0:
            logger.log('Training finished')

            if self.evaluate:
                tabular.record('Epoch', epoch)
                tabular.record('AverageReturn', np.mean(self.episode_rewards))
                tabular.record('StdReturn', np.std(self.episode_rewards))
                tabular.record('Policy/AveragePolicyLoss',
                               np.mean(self.episode_policy_losses))
                tabular.record('QFunction/AverageQFunctionLoss',
                               np.mean(self.episode_qf_losses))
                tabular.record('QFunction/AverageQ', np.mean(self.epoch_qs))
                tabular.record('QFunction/MaxQ', np.max(self.epoch_qs))
                tabular.record('QFunction/AverageAbsQ',
                               np.mean(np.abs(self.epoch_qs)))
                tabular.record('QFunction/AverageY', np.mean(self.epoch_ys))
                tabular.record('QFunction/MaxY', np.max(self.epoch_ys))
                tabular.record('QFunction/AverageAbsY',
                               np.mean(np.abs(self.epoch_ys)))
                tabular.record('AverageSuccessRate',
                               np.mean(self.success_history))

            if not self.smooth_return:
                self.episode_rewards = []
                self.episode_policy_losses = []
                self.episode_qf_losses = []
                self.epoch_ys = []
                self.epoch_qs = []

            self.success_history.clear()

        return last_average_return

    def optimize_policy(self, itr, samples):
        """
        Perform algorithm optimizing.

        Returns:
            action_loss: Loss of action predicted by the policy network.
            qval_loss: Loss of Q-value predicted by the Q-network.
            ys: y_s.
            qval: Q-value predicted by the Q-network.

        """
        transitions = np_to_torch(samples)
        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        terminals = transitions['terminal']

        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        next_inputs = next_observations
        inputs = observations
        with torch.no_grad():
            next_actions = self.target_policy(next_inputs)
            target_qvals = self.target_qf(next_inputs, next_actions)

        clip_range = (-self.clip_return,
                      0. if self.clip_pos_returns else self.clip_return)

        y_target = rewards + (1.0 - terminals) * self.discount * target_qvals
        y_target = torch.clamp(y_target, clip_range[0], clip_range[1])

        # optimize critic
        qval = self.qf(inputs, actions)
        qf_loss = torch.nn.MSELoss()
        qval_loss = qf_loss(qval, y_target)
        self.qf_optimizer.zero_grad()
        qval_loss.backward()
        self.qf_optimizer.step()

        # optimize actor
        actions = self.policy(inputs)
        action_loss = -1 * self.qf(inputs, actions).mean()
        self.policy_optimizer.zero_grad()
        action_loss.backward()
        self.policy_optimizer.step()

        # update target networks
        self.update_target()
        return (qval_loss.detach(), y_target, qval.detach(),
                action_loss.detach())

    def update_target(self):
        """Update parameters in the target policy and Q-value network."""
        for t_param, param in zip(self.target_qf.parameters(),
                                  self.qf.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                               param.data * self.tau)

        for t_param, param in zip(self.target_policy.parameters(),
                                  self.policy.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                               param.data * self.tau)
