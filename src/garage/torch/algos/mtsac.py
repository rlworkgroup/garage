"""This modules creates a MTSAC model in PyTorch."""
import numpy as np
import torch

from garage import EpisodeBatch, log_multitask_performance
from garage.np import obtain_evaluation_episodes
from garage.torch import global_device
from garage.torch.algos import SAC


class MTSAC(SAC):
    """A MTSAC Model in Torch.

    This MTSAC implementation uses is the same as SAC except for a small change
    called "disentangled alphas". Alpha is the entropy coefficient that is used
    to control exploration of the agent/policy. Disentangling alphas refers to
    having a separate alpha coefficients for every task learned by the policy.
    The alphas are accessed by using a the one-hot encoding of an id that is
    assigned to each task.

    Args:
        policy (garage.torch.policy.Policy): Policy/Actor/Agent that is being
            optimized by SAC.
        qf1 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        qf2 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        replay_buffer (ReplayBuffer): Stores transitions that are previously
            collected by the sampler.
        env_spec (EnvSpec): The env_spec attribute of the environment that the
            agent is being trained in.
        num_tasks (int): The number of tasks being learned.
        max_episode_length (int): The max episode length of the algorithm.
        eval_env (Environment): The environment used for collecting evaluation
            episodes.
        gradient_steps_per_itr (int): Number of optimization steps that should
            occur before the training step is over and a new batch of
            transitions is collected by the sampler.
        fixed_alpha (float): The entropy/temperature to be used if temperature
            is not supposed to be learned.
        target_entropy (float): target entropy to be used during
            entropy/temperature optimization. If None, the default heuristic
            from Soft Actor-Critic Algorithms and Applications is used.
        initial_log_entropy (float): initial entropy/temperature coefficient
            to be used if a fixed_alpha is not being used (fixed_alpha=None),
            and the entropy/temperature coefficient is being learned.
        discount (float): The discount factor to be used during sampling and
            critic/q_function optimization.
        buffer_batch_size (int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size (int): The minimum number of transitions that need to
            be in the replay buffer before training can begin.
        target_update_tau (float): A coefficient that controls the rate at
            which the target q_functions update over optimization iterations.
        policy_lr (float): Learning rate for policy optimizers.
        qf_lr (float): Learning rate for q_function optimizers.
        reward_scale (float): Reward multiplier. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer (torch.optim.Optimizer): Optimizer to be used for
            policy/actor, q_functions/critics, and temperature/entropy
            optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.
        num_evaluation_episodes (int): The number of evaluation episodes used
            for computing eval stats at the end of every epoch.

    """

    def __init__(
        self,
        policy,
        qf1,
        qf2,
        replay_buffer,
        env_spec,
        num_tasks,
        max_episode_length,
        eval_env,
        gradient_steps_per_itr,
        fixed_alpha=None,
        target_entropy=None,
        initial_log_entropy=0.,
        discount=0.99,
        buffer_batch_size=64,
        min_buffer_size=int(1e4),
        target_update_tau=5e-3,
        policy_lr=3e-4,
        qf_lr=3e-4,
        reward_scale=1.0,
        optimizer=torch.optim.Adam,
        steps_per_epoch=1,
        num_evaluation_episodes=5,
    ):

        super().__init__(policy=policy,
                         qf1=qf1,
                         qf2=qf2,
                         replay_buffer=replay_buffer,
                         env_spec=env_spec,
                         max_episode_length=max_episode_length,
                         gradient_steps_per_itr=gradient_steps_per_itr,
                         fixed_alpha=fixed_alpha,
                         target_entropy=target_entropy,
                         initial_log_entropy=initial_log_entropy,
                         discount=discount,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         target_update_tau=target_update_tau,
                         policy_lr=policy_lr,
                         qf_lr=qf_lr,
                         reward_scale=reward_scale,
                         optimizer=optimizer,
                         steps_per_epoch=steps_per_epoch,
                         num_evaluation_episodes=num_evaluation_episodes,
                         eval_env=eval_env)
        self._num_tasks = num_tasks
        self._eval_env = eval_env
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                self._target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            self._log_alpha = torch.Tensor([self._initial_log_entropy] *
                                           self._num_tasks).requires_grad_()
            self._alpha_optimizer = optimizer([self._log_alpha] *
                                              self._num_tasks,
                                              lr=self._policy_lr)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha] *
                                           self._num_tasks).log()
        self._epoch_mean_success_rate = []
        self._epoch_median_success_rate = []

    def _get_log_alpha(self, samples_data):
        """Return the value of log_alpha.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: log_alpha. shape is (1, self.buffer_batch_size)

        """
        obs = samples_data['observation']
        log_alpha = self._log_alpha
        one_hots = obs[:, -self._num_tasks:]
        ret = torch.mm(one_hots, log_alpha.unsqueeze(0).t()).squeeze()
        return ret

    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """
        eval_eps = []
        for _ in range(self._num_tasks):
            eval_eps.append(
                obtain_evaluation_episodes(
                    self.policy,
                    self._eval_env,
                    num_eps=self._num_evaluation_episodes))
        eval_eps = EpisodeBatch.concatenate(*eval_eps)
        last_return = log_multitask_performance(epoch, eval_eps,
                                                self._discount)
        return last_return

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        super().to(device)
        if device is None:
            device = global_device()
        if not self._use_automatic_entropy_tuning:
            self._log_alpha = torch.Tensor([self._fixed_alpha] *
                                           self._num_tasks).log().to(device)
        else:
            self._log_alpha = torch.Tensor(
                [self._initial_log_entropy] *
                self._num_tasks).to(device).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                                    lr=self._policy_lr)
