"""This modules creates a DDPG model in PyTorch."""
import collections
import copy

from dowel import logger, tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import _Default, log_performance, make_optimizer
from garage._functions import obtain_evaluation_episodes
from garage.np.algos import RLAlgorithm
from garage.torch import global_device, np_to_torch
from garage.torch._functions import zero_optim_grads


class DQN(RLAlgorithm):
    """DQN algorithm. See https://arxiv.org/pdf/1312.5602.pdf.

    DQN, also known as the Deep Q Network algorithm, is an off-policy algorithm
    that learns action-value estimates for each state, action pair. The
    policy then simply acts by taking the action that yields the highest Q(s,a)
    value for a given state s.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy. For DQN, this is a
            policy that performs the action that yields the highest Q value.
        qf (nn.Module): Q-value network.
        replay_buffer (ReplayBuffer): Replay buffer.
        sampler (garage.sampler.Sampler): Sampler.
        steps_per_epoch (int): Number of train_once calls per epoch.
        n_train_steps (int): Training steps.
        eval_env (Environment): Evaluation environment. If None, a copy of the
            main environment is used for evaluation.
        double_q (bool): Whether to use Double DQN.
            See https://arxiv.org/abs/1509.06461.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If `None`, defaults to
            `env_spec.max_episode_length`.
        buffer_batch_size (int): Batch size of replay buffer.
        min_buffer_size (int): The minimum buffer size for replay buffer.
        exploration_policy (ExplorationPolicy): Exploration strategy, typically
            epsilon-greedy.
        num_eval_episodes (int): Nunber of evaluation episodes. Defaults to 10.
        deterministic_eval (bool): Whether to evaluate the policy
            deterministically (without exploration noise). False by default.
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
        clip_gradient (float): Clip gradient norm to `clip_gradient`. If None,
            gradient are not clipped. Defaults to 10.
        reward_scale (float): Reward scale.
    """

    def __init__(
            self,
            env_spec,
            policy,
            qf,
            replay_buffer,
            sampler,
            exploration_policy=None,
            eval_env=None,
            double_q=True,
            qf_optimizer=torch.optim.Adam,
            *,  # Everything after this is numbers.
            steps_per_epoch=20,
            n_train_steps=50,
            max_episode_length_eval=None,
            deterministic_eval=False,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            num_eval_episodes=10,
            discount=0.99,
            qf_lr=_Default(1e-3),
            clip_rewards=None,
            clip_gradient=10,
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
        self._double_q = double_q
        self._discount = discount
        self._reward_scale = reward_scale
        self.max_episode_length = env_spec.max_episode_length
        self._max_episode_length_eval = (max_episode_length_eval
                                         or self.max_episode_length)
        self._episode_reward_mean = collections.deque(maxlen=100)
        self._num_eval_episodes = num_eval_episodes
        self._deterministic_eval = deterministic_eval

        self.env_spec = env_spec
        self.replay_buffer = replay_buffer
        self.policy = policy
        self.exploration_policy = exploration_policy

        self._target_qf = copy.deepcopy(self._qf)
        self._qf_optimizer = make_optimizer(qf_optimizer,
                                            module=self._qf,
                                            lr=qf_lr)
        self._eval_env = eval_env

        self._sampler = sampler

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_returns = [float('nan')]

        if self._min_buffer_size > self.replay_buffer.n_transitions_stored:
            num_warmup_steps = (self._min_buffer_size -
                                self.replay_buffer.n_transitions_stored)
            self.replay_buffer.add_episode_batch(
                trainer.obtain_episodes(0, num_warmup_steps))

        trainer.enable_logging = True

        for _ in trainer.step_epochs():
            if (self.replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                logger.log('Evaluating policy')

                params_before = self.exploration_policy.get_param_values()
                eval_eps = obtain_evaluation_episodes(
                    (self.exploration_policy
                     if not self._deterministic_eval else self.policy),
                    self._eval_env,
                    num_eps=self._num_eval_episodes,
                    max_episode_length=self._max_episode_length_eval)
                self.exploration_policy.set_param_values(params_before)

                last_returns = log_performance(trainer.step_itr,
                                               eval_eps,
                                               discount=self._discount)
                self._episode_reward_mean.extend(last_returns)
                tabular.record('Evaluation/100EpRewardMean',
                               np.mean(self._episode_reward_mean))

            for _ in range(self._steps_per_epoch):
                trainer.step_episode = trainer.obtain_episodes(
                    trainer.step_itr)
                if hasattr(self.exploration_policy, 'update'):
                    self.exploration_policy.update(trainer.step_episode)

                self._train_once(trainer.step_itr, trainer.step_episode)
                trainer.step_itr += 1

        return np.mean(last_returns)

    def _train_once(self, itr, episodes):
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
                timesteps = self.replay_buffer.sample_timesteps(
                    self._buffer_batch_size)
                qf_loss, y, q = tuple(v.cpu().numpy()
                                      for v in self._optimize_qf(timesteps))

                self._episode_qf_losses.append(qf_loss)
                self._epoch_ys.append(y)
                self._epoch_qs.append(q)

        if itr % self._steps_per_epoch == 0:
            self._log_eval_results(epoch)

        if itr % self._target_update_freq == 0:
            self._target_qf = copy.deepcopy(self._qf)

    def _log_eval_results(self, epoch):
        """Log evaluation results after an epoch.

        Args:
            epoch (int): Current epoch.
        """
        logger.log('Training finished')

        if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
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

    def _optimize_qf(self, timesteps):
        """Perform algorithm optimizing.

        Args:
            timesteps (TimeStepBatch): Processed batch data.

        Returns:
            qval_loss: Loss of Q-value predicted by the Q-network.
            ys: y_s.
            qval: Q-value predicted by the Q-network.

        """
        observations = np_to_torch(timesteps.observations)
        rewards = np_to_torch(timesteps.rewards).reshape(-1, 1)
        rewards *= self._reward_scale
        actions = np_to_torch(timesteps.actions)
        next_observations = np_to_torch(timesteps.next_observations)
        terminals = np_to_torch(timesteps.terminals).reshape(-1, 1)

        next_inputs = next_observations
        inputs = observations
        with torch.no_grad():
            if self._double_q:
                # Use online qf to get optimal actions
                selected_actions = torch.argmax(self._qf(next_inputs), axis=1)
                # use target qf to get Q values for those actions
                selected_actions = selected_actions.long().unsqueeze(1)
                best_qvals = torch.gather(self._target_qf(next_inputs),
                                          dim=1,
                                          index=selected_actions)
            else:
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

        zero_optim_grads(self._qf_optimizer)
        qval_loss.backward()

        # optionally clip the gradients
        if self._clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                           self._clip_grad)
        self._qf_optimizer.step()

        return (qval_loss.detach(), y_target, selected_qs.detach())

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()
        logger.log('Using device: ' + str(device))
        self._qf = self._qf.to(device)
        self._target_qf = self._target_qf.to(device)
