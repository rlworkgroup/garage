"""This modules creates a sac model in PyTorch."""
from collections import deque
import copy

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import log_performance
from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
import garage.torch.utils as tu


class SAC(OffPolicyRLAlgorithm):
    """A SAC Model in Torch.

    Based on Soft Actor-Critic and Applications:
        https://arxiv.org/abs/1812.05905

    Soft Actor-Critic (SAC) is an algorithm which optimizes a stochastic
    policy in an off-policy way, forming a bridge between stochastic policy
    optimization and DDPG-style approaches.
    A central feature of SAC is entropy regularization. The policy is trained
    to maximize a trade-off between expected return and entropy, a measure of
    randomness in the policy. This has a close connection to the
    exploration-exploitation trade-off: increasing entropy results in more
    exploration, which can accelerate learning later on. It can also prevent
    the policy from prematurely converging to a bad local optimum.

    Args:
        policy(garage.torch.policy): Policy/Actor/Agent that is being optimized
            by SAC.
        qf1(garage.torch.q_function): QFunction/Critic used for actor/policy
            optimization. See Soft Actor-Critic and Applications.
        qf2(garage.torch.q_function): QFunction/Critic used for actor/policy
            optimization. See Soft Actor-Critic and Applications.
        replay_buffer(garage.replay_buffer): Stores transitions that
            are previously collected by the sampler.
        env_spec(garage.envs.env_spec.EnvSpec): The env_spec attribute of the
            environment that the agent is being trained in. Usually accessable
            by calling env.spec.
        max_path_length(int): Max path length of the environment.
        gradient_steps_per_itr(int): Number of optimization steps that should
            occur before the training step is over and a new batch of
            transitions is collected by the sampler.
        use_automatic_entropy_tuning(bool): True if the entropy/temperature
            coefficient should be learned. False if it should be static.
        alpha(float): entropy/temperature to be used if
            `use_automatic_entropy_tuning` is False.
        target_entropy(float): target entropy to be used during
            entropy/temperature optimization. If None, the default heuristic
            from Soft Actor-Critic Algorithms and Applications is used.
        initial_log_entropy(float): initial entropy/temperature coefficient
            to be used if use_automatic_entropy_tuning is True.
        discount(float): Discount factor to be used during sampling and
            critic/q_function optimization.
        buffer_batch_size(int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size(int): The minimum number of transitions that need to be
            in the replay buffer before training can begin.
        target_update_tau(float): coefficient that controls the rate at which
            the target q_functions update over optimization iterations.
        policy_lr(float): learning rate for policy optimizers.
        qf_lr(float): learning rate for q_function optimizers.
        reward_scale (float): reward scale. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer(torch.optim): optimizer to be used for policy/actor,
            q_functions/critics, and temperature/entropy optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.

    """

    def __init__(
            self,
            policy,
            qf1,
            qf2,
            replay_buffer,
            env_spec,
            max_path_length,
            gradient_steps_per_itr,
            use_automatic_entropy_tuning=True,
            alpha=None,
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
    ):

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.replay_buffer = replay_buffer
        self.tau = target_update_tau
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self._initial_log_entropy = initial_log_entropy
        self._gradient_steps = gradient_steps_per_itr
        self._optimizer = optimizer

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         qf=qf1,
                         n_train_steps=self._gradient_steps,
                         max_path_length=max_path_length,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         replay_buffer=replay_buffer,
                         use_target=True,
                         discount=discount,
                         steps_per_epoch=steps_per_epoch)
        self.reward_scale = reward_scale
        # use 2 target q networks
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.policy_optimizer = self._optimizer(self.policy.parameters(),
                                                lr=self.policy_lr)
        self.qf1_optimizer = self._optimizer(self.qf1.parameters(),
                                             lr=self.qf_lr)
        self.qf2_optimizer = self._optimizer(self.qf2.parameters(),
                                             lr=self.qf_lr)
        # automatic entropy coefficient tuning
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning and not alpha:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            self.log_alpha = torch.Tensor([self._initial_log_entropy
                                           ]).requires_grad_()
            self.alpha_optimizer = optimizer([self.log_alpha],
                                             lr=self.policy_lr)
        else:
            self.log_alpha = torch.Tensor([alpha]).log()
        self.episode_rewards = deque(maxlen=30)

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
        for _ in runner.step_epochs():
            for _ in range(self.steps_per_epoch):
                if not self._buffer_prefilled:
                    batch_size = int(self.min_buffer_size)
                else:
                    batch_size = None
                runner.step_path = runner.obtain_samples(
                    runner.step_itr, batch_size)
                path_returns = []
                for path in runner.step_path:
                    self.replay_buffer.add_transitions(
                        observation=path['observations'],
                        action=path['actions'],
                        reward=path['rewards'],
                        next_observation=path['next_observations'],
                        terminal=path['dones'])
                    path_returns.append(sum(path['rewards']))
                assert len(path_returns) is len(runner.step_path)
                self.episode_rewards.append(np.mean(path_returns))
                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss = self.train_once()
            last_return = log_performance(runner.step_itr,
                                          self._obtain_evaluation_samples(
                                              runner.get_env_copy(),
                                              num_trajs=10),
                                          discount=self.discount)
            self._log_statistics(policy_loss, qf1_loss, qf2_loss)
            tabular.record('TotalEnvSteps', runner.total_env_steps)
            runner.step_itr += 1

        return np.mean(last_return)

    def train_once(self, itr=None, paths=None):
        """Complete 1 training iteration of SAC.

        Args:
            itr (int): Iteration number. This argument is deprecated.
            paths (list[dict]): A list of collected paths.
                This argument is deprecated.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        del itr
        del paths
        if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:
            samples = self.replay_buffer.sample(self.buffer_batch_size)
            samples = tu.dict_np_to_torch(samples)
            policy_loss, qf1_loss, qf2_loss = self.optimize_policy(0, samples)
            self._update_targets()

        return policy_loss, qf1_loss, qf2_loss

    def _get_log_alpha(self, **kwargs):
        """Return the value of log_alpha.

        This function exists in case there are versions of sac that need
        access to a modified log_alpha, such as multi_task sac.

        Args:
            kwargs(dict): keyword args that can be used in retrieving the
                log_alpha parameter. Unused here.

        Returns:
            torch.Tensor: log_alpha

        """
        del kwargs
        log_alpha = self.log_alpha
        return log_alpha

    def _temperature_objective(self, log_pi):
        """Compute the temperature/alpha coefficient loss.

        Args:
            log_pi(torch.Tensor): log probability of actions that are sampled
                from the replay buffer. Shape is (1, buffer_batch_size).

        Returns:
            torch.Tensor: the temperature/alpha coefficient loss.

        """
        alpha_loss = 0
        if self.use_automatic_entropy_tuning:
            alpha_loss = (-(self._get_log_alpha()) *
                          (log_pi.detach() + self.target_entropy)).mean()
        return alpha_loss

    def _actor_objective(self, obs, new_actions, log_pi_new_actions):
        """Compute the Policy/Actor loss.

        Args:
            obs(torch.Tensor): Observations from transitions sampled from the
                replay buffer. Shape is (observation_dim, buffer_batch_size).
            new_actions(torch.Tensor): Actions resampled from the policy based
                based on the Observations, obs, which were sampled from the
                replay buffer. Shape is (action_dim, buffer_batch_size).
            log_pi_new_actions(torch.Tensor): Log probability of the new
                actions on the TanhNormal distributions that they were sampled
                from. Shape is (1, buffer_batch_size).

        Returns:
            torch.Tensor: loss from the Policy/Actor.

        """
        with torch.no_grad():
            alpha = self._get_log_alpha().exp()
        min_q_new_actions = torch.min(self.qf1(obs, new_actions),
                                      self.qf2(obs, new_actions))
        policy_objective = ((alpha * log_pi_new_actions) -
                            min_q_new_actions.flatten()).mean()
        return policy_objective

    def _critic_objective(self, samples):
        """Compute the Q-function/critic loss.

        Args:
            samples(dict): Transitions that are sampled from the replay buffer.

        Returns:
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples['observation']
        actions = samples['action']
        rewards = samples['reward']
        terminals = samples['terminal']
        next_obs = samples['next_observation']
        with torch.no_grad():
            alpha = self._get_log_alpha().exp()

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        new_next_actions_dist = self.policy(next_obs)
        new_next_actions_pre_tanh, new_next_actions = (
            new_next_actions_dist.rsample_with_pre_tanh_value())
        new_log_pi = new_next_actions_dist.log_prob(
            value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)

        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs,
                            new_next_actions)).flatten() - (alpha * new_log_pi)
        with torch.no_grad():
            q_target = rewards + (1. -
                                  terminals) * self.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        return qf1_loss, qf2_loss

    def _update_targets(self):
        """Update parameters in the target q-functions."""
        target_qfs = [self.target_qf1, self.target_qf2]
        qfs = [self.qf1, self.qf2]
        for target_qf, qf in zip(target_qfs, qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                   param.data * self.tau)

    def optimize_policy(self, itr, samples_data):
        """Optimize the policy q_functions, and temperature coefficient.

        Args:
            itr (int): Iterations.
            samples_data (list): Processed batch data.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        qf1_loss, qf2_loss = self._critic_objective(samples_data)

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        action_dists = self.policy(obs)
        new_actions_pre_tanh, new_actions = (
            action_dists.rsample_with_pre_tanh_value())
        log_pi_new_actions = action_dists.log_prob(
            value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        policy_loss = self._actor_objective(obs, new_actions,
                                            log_pi_new_actions)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        self.policy_optimizer.step()

        if self.use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        return policy_loss, qf1_loss, qf2_loss

    def _log_statistics(self, policy_loss, qf1_loss, qf2_loss):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss(torch.Tensor): loss from actor/policy network.
            qf1_loss(torch.Tensor): loss from 1st qf/critic network.
            qf2_loss(torch.Tensor): loss from 2nd qf/critic network.

        """
        with torch.no_grad():
            tabular.record('alpha', self.log_alpha.exp().item())
        tabular.record('policy_loss', policy_loss.item())
        tabular.record('qf_loss/{}'.format('qf1_loss'), float(qf1_loss))
        tabular.record('qf_loss/{}'.format('qf2_loss'), float(qf2_loss))
        tabular.record('buffer_size', self.replay_buffer.n_transitions_stored)
        tabular.record('local/average_return', np.mean(self.episode_rewards))

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2
        ]

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = tu.global_device()
        for net in self.networks:
            net.to(device)
        self.log_alpha = torch.Tensor([self._initial_log_entropy
                                       ]).to(device).requires_grad_()
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer = self._optimizer([self.log_alpha],
                                                   lr=self.policy_lr)
