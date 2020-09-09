"""This modules creates a DDPG model in PyTorch."""
import copy

from dowel import logger, tabular
import numpy as np
import torch
import collections
import torch.nn.functional as F

from garage import _Default, log_performance, make_optimizer
from garage._functions import obtain_evaluation_episodes
from garage.np.algos import RLAlgorithm
from garage.sampler import DefaultWorker, FragmentWorker
from garage.torch import dict_np_to_torch, global_device


class DQN(RLAlgorithm):
    """A DDPG model implemented with PyTorch.

    DDPG, also known as Deep Deterministic Policy Gradient, uses actor-critic
    method to optimize the policy and Q-function prediction. It uses a
    supervised method to update the critic network and policy gradient to
    update the actor network. And there are exploration strategy, replay
    buffer and target networks involved to stabilize the training process.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy. For DQN, this is a
            policy that performs the action that yields the highest Q value.
        qf (object): Q-value network.
        replay_buffer (ReplayBuffer): Replay buffer.
        steps_per_epoch (int): Number of train_once calls per epoch.
        n_train_steps (int): Training steps.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If `None`, defaults to
            `env_spec.max_episode_length`.
        buffer_batch_size (int): Batch size of replay buffer.
        min_buffer_size (int): The minimum buffer size for replay buffer.
        exploration_policy (ExplorationPolicy): Exploration strategy, typically
            epsilon-greedy.
        target_update_freq (int): Number of optimization steps between each
            update to the target Q network.
        discount(float): Discount factor for the cumulative return.
        qf_optimizer (Union[type, tuple[type, dict]]): Type of optimizer
            for training Q-value network. This can be an optimizer type such
            as `torch.optim.Adam` or a tuple of type and dictionary, where
            dictionary contains arguments to initialize the optimizer
            e.g. `(torch.optim.Adam, {'lr' : 1e-3})`.
        qf_lr (float): Learning rate for Q-value network parameters.
        clip_rewards (float): Clip reward to be in [-clip_rewards,
            clip_rewards]. If None, rewards are not clipped.
        clip_gradient (float): Clip gradients to be in
            [-clip_gradient, clip_gradient]. If None, gradients are not
            clipped. Defaults to 1.0.
        reward_scale (float): Reward scale.
    """

    def __init__(
            self,
            env_spec,
            policy,
            qf,
            replay_buffer,
            exploration_policy=None,
            qf_optimizer=torch.optim.Adam,
            *,  # Everything after this is numbers.
            steps_per_epoch=20,
            n_train_steps=50,
            max_episode_length_eval=None,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            discount=0.99,
            qf_lr=_Default(1e-3),
            clip_rewards=None,
            clip_gradient=1.0,
            target_update_freq=5,
            reward_scale=1.):
        self._clip_reward = clip_rewards
        self._clip_grad = clip_gradient

        self._steps_per_epoch = steps_per_epoch
        self._target_update_freq = target_update_freq
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
        self.max_episode_length = env_spec.max_episode_length
        self._max_episode_length_eval = (max_episode_length_eval
                                         or self.max_episode_length)
        self._episode_reward_mean = collections.deque(maxlen=100)

        self.env_spec = env_spec
        self.replay_buffer = replay_buffer
        self.policy = policy
        self.exploration_policy = exploration_policy

        self._target_qf = copy.deepcopy(self._qf)
        self._qf_optimizer = make_optimizer(qf_optimizer,
                                            module=self._qf,
                                            lr=qf_lr)
        self._eval_env = None
        self.worker_cls = FragmentWorker

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): Experiment runner.

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
                    logger.log('Evaluating policy')

                    eval_eps = obtain_evaluation_episodes(
                        self.policy,
                        self._eval_env,
                        num_eps=10,
                        max_episode_length=self._max_episode_length_eval)
                    last_returns = log_performance(runner.step_itr,
                                                   eval_eps,
                                                   discount=self._discount)
                    self._episode_reward_mean.extend(last_returns)
                    tabular.record('Evaluation/100EpRewardMean',
                                   np.mean(self._episode_reward_mean))
                runner.step_itr += 1

        return np.mean(last_returns)

    def train_once(self, itr, episodes):
        """Perform one iteration of training.

        Args:
            itr (int): Iteration number.
            episodes (EpisodeBatch): Batch of episodes.

        """
        self.replay_buffer.add_episode_batch(episodes)

        epoch = itr / self._steps_per_epoch

        for _ in range(self._n_train_steps):
            if (self.replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                samples = self.replay_buffer.sample_transitions(
                    self._buffer_batch_size)
                samples['rewards'] *= self._reward_scale
                qf_loss, y, q = tuple(v.cpu().numpy()
                                      for v in self.optimize_qf(samples))

                self._episode_qf_losses.append(qf_loss)
                self._epoch_ys.append(y)
                self._epoch_qs.append(q)

        if itr % self._steps_per_epoch == 0:
            logger.log('Training finished')

            if (self.replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                tabular.record('Epoch', epoch)
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

        if itr % self._target_update_freq == 0:
            self._target_qf = copy.deepcopy(self._qf)

    def optimize_qf(self, samples_data):
        """Perform algorithm optimizing.

        Args:
            samples_data (dict): Processed batch data.

        Returns:
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
            # discrete, outputs Qs for all possible actions
            target_qvals = self._target_qf(next_inputs)
            best_qvals, _ = torch.max(target_qvals, 1)
            best_qvals = best_qvals.unsqueeze(1)

        rewards_clipped = rewards
        if self._clip_reward is not None:
            rewards_clipped = torch.clamp(rewards, -1 * self._clip_reward,
                                          self._clip_reward)
        y_target = (rewards_clipped +
                    (1.0 - terminals) * self._discount * best_qvals)
        y_target = y_target.squeeze(1)

        # optimize qf
        qvals = self._qf(inputs)
        selected_qs = torch.sum(qvals * actions, axis=1)
        qval_loss = F.smooth_l1_loss(selected_qs, y_target)

        self._qf_optimizer.zero_grad()
        qval_loss.backward()

        # optionally clip the gradients
        if self._clip_grad is not None:
            for param in self._qf.parameters():
                param.grad.data.clamp_(-self._clip_grad, self._clip_grad)
        self._qf_optimizer.step()

        return (qval_loss.detach(), y_target, selected_qs.detach())

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()
        print(device)
        print(torch.cuda.is_available())
        self._qf = self._qf.to(device)
        self._target_qf = self._target_qf.to(device)
