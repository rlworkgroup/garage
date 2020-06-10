"""This modules creates a DDPG model in PyTorch."""
from collections import deque
import copy

from dowel import logger, tabular
import numpy as np
import torch

from garage import _Default, make_optimizer
from garage import log_performance
from garage.np import obtain_evaluation_samples
from garage.np import samples_to_tensors
from garage.np.algos import RLAlgorithm
from garage.sampler import OffPolicyVectorizedSampler
from garage.torch import dict_np_to_torch, torch_to_np


class DDPG(RLAlgorithm):
    """A DDPG model implemented with PyTorch.

    DDPG, also known as Deep Deterministic Policy Gradient, uses actor-critic
    method to optimize the policy and Q-function prediction. It uses a
    supervised method to update the critic network and policy gradient to
    update the actor network. And there are exploration strategy, replay
    buffer and target networks involved to stabilize the training process.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        qf (object): Q-value network.
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
        exploration_policy (garage.np.exploration_policies.ExplorationPolicy): # noqa: E501
                Exploration strategy.
        target_update_tau (float): Interpolation parameter for doing the
            soft target update.
        discount(float): Discount factor for the cumulative return.
        policy_weight_decay (float): L2 weight decay factor for parameters
            of the policy network.
        qf_weight_decay (float): L2 weight decay factor for parameters
            of the q value network.
        policy_optimizer (Union[type, tuple[type, dict]]): Type of optimizer
            for training policy network. This can be an optimizer type such as
            `torch.optim.Adam` or a tuple of type and dictionary, where
            dictionary contains arguments to initialize the optimizer
            e.g. `(torch.optim.Adam, {'lr' : 1e-3})`.
        qf_optimizer (Union[type, tuple[type, dict]]): Type of optimizer
            for training Q-value network. This can be an optimizer type such
            as `torch.optim.Adam` or a tuple of type and dictionary, where
            dictionary contains arguments to initialize the optimizer
            e.g. `(torch.optim.Adam, {'lr' : 1e-3})`.
        policy_lr (float): Learning rate for policy network parameters.
        qf_lr (float): Learning rate for Q-value network parameters.
        clip_pos_returns (bool): Whether or not clip positive returns.
        clip_return (float): Clip return to be in [-clip_return,
            clip_return].
        max_action (float): Maximum action magnitude.
        reward_scale (float): Reward scale.
        smooth_return (bool): Whether to smooth the return for logging.

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
            policy_optimizer=torch.optim.Adam,
            qf_optimizer=torch.optim.Adam,
            policy_lr=_Default(1e-4),
            qf_lr=_Default(1e-3),
            clip_pos_returns=False,
            clip_return=np.inf,
            max_action=None,
            reward_scale=1.,
            smooth_return=True):
        action_bound = env_spec.action_space.high
        self._tau = target_update_tau
        self._policy_weight_decay = policy_weight_decay
        self._qf_weight_decay = qf_weight_decay
        self._clip_pos_returns = clip_pos_returns
        self._clip_return = clip_return
        self._max_action = action_bound if max_action is None else max_action

        self._steps_per_epoch = steps_per_epoch
        self._success_history = deque(maxlen=100)
        self._episode_rewards = []
        self._episode_policy_losses = []
        self._episode_qf_losses = []
        self._epoch_ys = []
        self._epoch_qs = []

        self._policy = policy
        self._qf = qf
        self._n_train_steps = n_train_steps

        self._min_buffer_size = min_buffer_size
        self._qf = qf
        self._steps_per_epoch = steps_per_epoch
        self._n_train_steps = n_train_steps
        self._buffer_batch_size = buffer_batch_size
        self._discount = discount
        self._reward_scale = reward_scale
        self._smooth_return = smooth_return
        self.max_path_length = max_path_length
        self._max_eval_path_length = max_eval_path_length

        # used by OffPolicyVectorizedSampler
        self.env_spec = env_spec
        self.rollout_batch_size = rollout_batch_size
        self.replay_buffer = replay_buffer
        self.policy = policy
        self.exploration_policy = exploration_policy

        self._target_policy = copy.deepcopy(self.policy)
        self._target_qf = copy.deepcopy(self._qf)
        self._policy_optimizer = make_optimizer(policy_optimizer,
                                                module=self.policy,
                                                lr=policy_lr)
        self._qf_optimizer = make_optimizer(qf_optimizer,
                                            module=self._qf,
                                            lr=qf_lr)

        self.sampler_cls = OffPolicyVectorizedSampler

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None
        runner.enable_logging = False

        for _ in runner.step_epochs():
            for cycle in range(self._steps_per_epoch):
                runner.step_path = runner.obtain_samples(runner.step_itr)
                for path in runner.step_path:
                    path['rewards'] *= self._reward_scale
                last_return = self.train_once(runner.step_itr,
                                              runner.step_path)
                if (cycle == 0 and self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    runner.enable_logging = True
                    log_performance(runner.step_itr,
                                    obtain_evaluation_samples(
                                        self.policy, runner.get_env_copy()),
                                    discount=self._discount)
                runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        """Perform one iteration of training.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        Returns:
            float: Average return.

        """
        paths = samples_to_tensors(paths)

        epoch = itr / self._steps_per_epoch

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
            if (itr % self._steps_per_epoch == 0
                    and (self.replay_buffer.n_transitions_stored >=
                         self._min_buffer_size)):
                avg_success_rate = np.mean(self._success_history)

        for _ in range(self._n_train_steps):
            if (self.replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                samples = self.replay_buffer.sample_transitions(
                    self._buffer_batch_size)
                qf_loss, y, q, policy_loss = torch_to_np(
                    self.optimize_policy(samples))

                self._episode_policy_losses.append(policy_loss)
                self._episode_qf_losses.append(qf_loss)
                self._epoch_ys.append(y)
                self._epoch_qs.append(q)

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
                tabular.record('AverageSuccessRate', avg_success_rate)

            if not self._smooth_return:
                self._episode_rewards = []
                self._episode_policy_losses = []
                self._episode_qf_losses = []
                self._epoch_ys = []
                self._epoch_qs = []

            self._success_history.clear()

        return last_average_return

    def optimize_policy(self, samples_data):
        """Perform algorithm optimizing.

        Args:
            samples_data (dict): Processed batch data.

        Returns:
            action_loss: Loss of action predicted by the policy network.
            qval_loss: Loss of Q-value predicted by the Q-network.
            ys: y_s.
            qval: Q-value predicted by the Q-network.

        """
        transitions = dict_np_to_torch(samples_data)

        observations = transitions['observations']
        rewards = transitions['rewards'].reshape(-1, 1)
        actions = transitions['actions']
        next_observations = transitions['next_observations']
        terminals = transitions['terminals'].reshape(-1, 1)

        next_inputs = next_observations
        inputs = observations
        with torch.no_grad():
            next_actions = self._target_policy(next_inputs)
            target_qvals = self._target_qf(next_inputs, next_actions)

        clip_range = (-self._clip_return,
                      0. if self._clip_pos_returns else self._clip_return)

        y_target = rewards + (1.0 - terminals) * self._discount * target_qvals
        y_target = torch.clamp(y_target, clip_range[0], clip_range[1])

        # optimize critic
        qval = self._qf(inputs, actions)
        qf_loss = torch.nn.MSELoss()
        qval_loss = qf_loss(qval, y_target)
        self._qf_optimizer.zero_grad()
        qval_loss.backward()
        self._qf_optimizer.step()

        # optimize actor
        actions = self.policy(inputs)
        action_loss = -1 * self._qf(inputs, actions).mean()
        self._policy_optimizer.zero_grad()
        action_loss.backward()
        self._policy_optimizer.step()

        # update target networks
        self.update_target()
        return (qval_loss.detach(), y_target, qval.detach(),
                action_loss.detach())

    def update_target(self):
        """Update parameters in the target policy and Q-value network."""
        for t_param, param in zip(self._target_qf.parameters(),
                                  self._qf.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                               param.data * self._tau)

        for t_param, param in zip(self._target_policy.parameters(),
                                  self.policy.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                               param.data * self._tau)
