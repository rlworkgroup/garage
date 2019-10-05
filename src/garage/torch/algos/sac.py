"""This modules creates a sac model in PyTorch."""
import copy

from dowel import logger, tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.torch.utils import np_to_torch, torch_to_np


class SAC(OffPolicyRLAlgorithm):
    """ A SAC Model in Torch.

    Soft Actor Critic (SAC) is an algorithm which optimizes a stochastic
    policy in an off-policy way, forming a bridge between stochastic policy
    optimization and DDPG-style approaches.
    A central feature of SAC is entropy regularization. The policy is trained
    to maximize a trade-off between expected return and entropy, a measure of
    randomness in the policy. This has a close connection to the
    exploration-exploitation trade-off: increasing entropy results in more
    exploration, which can accelerate learning later on. It can also prevent
    the policy from prematurely converging to a bad local optimum.
    """

    def __init__(self,
                 env_spec,
                 policy,
                 qf1,
                 qf2,
                 replay_buffer,
                 gradient_steps_per_itr=1,
                 target_entropy=None,
                 use_automatic_entropy_tuning=False,
                 discount=0.99,
                 max_path_length=None,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 rollout_batch_size=1,
                 exploration_strategy=None,
                 target_update_tau=1e-2,
                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 policy_weight_decay=0,
                 qf_weight_decay=0,
                 optimizer=torch.optim.Adam,
                 clip_pos_returns=False,
                 clip_return=np.inf,
                 max_action=None,
                 smooth_return=True,
                 input_include_goal=False):

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.replay_buffer = replay_buffer
        self.tau = target_update_tau
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.gradient_steps = gradient_steps_per_itr
        self.policy_weight_decay = policy_weight_decay
        self.qf_weight_decay = qf_weight_decay
        self.clip_pos_returns = clip_pos_returns
        self.clip_return = clip_return
        self.evaluate = False
        self.input_include_goal = input_include_goal

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         qf=qf1,
                         n_train_steps=self.gradient_steps,
                         n_epoch_cycles=1,
                         max_path_length=max_path_length,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         rollout_batch_size=rollout_batch_size,
                         exploration_strategy=exploration_strategy,
                         replay_buffer=replay_buffer,
                         use_target=True,
                         discount=discount,
                         smooth_return=smooth_return)

        self.target_policy = copy.deepcopy(self.policy)
        # use 2 target q networks
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.policy_optimizer = optimizer(self.policy.parameters(),
                                          lr=self.policy_lr)
        self.qf1_optimizer = optimizer(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optimizer(self.qf2.parameters(), lr=self.qf_lr)

        # automatic entropy coefficient tuning
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
                self.log_alpha = torch.zeros(1, dtype=torch.float, requires_grad=True)
                self.alpha_optimizer = optimizer([self.log_alpha], lr=self.policy_lr)
        else:
            self.log_alpha = 0


        self.episode_rewards = []
        self.success_history = []

    # 0) update policy using updated min q function
    # 1) compute targets from Q functions
    # 2) update Q functions using optimizer
    # 3) query Q functons, take min of those functions
    def train_once(self, itr, paths):
        """
        """
        paths = self.process_samples(itr, paths)
        self.episode_rewards.extend([
            path for path, complete in zip(paths['undiscounted_returns'],
                                           paths['complete']) if complete
        ])
        self.success_history.extend([
            path for path, complete in zip(paths['success_history'],
                                           paths['complete']) if complete
        ])
        last_average_return = np.mean(self.episode_rewards)
        if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
            for gradient_step in range(self.gradient_steps):
                    samples = self.replay_buffer.sample(self.buffer_batch_size)
                    self.update_q_functions(itr, samples)
                    self.optimize_policy(itr, samples)
                    self.update_targets()
            tabular.record('reward', last_average_return)

        return last_average_return

    def update_q_functions(self, itr, samples):
        """ Update the q functions using the target q_functions.

        Args:
            itr (int) - current training iteration
            samples() - samples recovered from the replay buffer
        """
        obs = samples["observation"]
        actions = samples["action"]
        rewards = samples["reward"]
        next_obs = samples["next_observation"]

        with torch.no_grad():
            next_actions, _ = self.policy.get_actions(torch.Tensor(next_obs))
            next_ll = self.policy.log_likelihood(torch.Tensor(next_obs),
                                            torch.Tensor(next_actions))

        qfs = [self.qf1, self.qf2]
        target_qfs = [self.target_qf1, self.target_qf2]
        qf_optimizers = [self.qf1_optimizer, self.qf2_optimizer]
        qf_loss = []
        for target_qf, qf, qf_optimizer in zip(target_qfs, qfs, qf_optimizers):
            curr_q_val = qf(torch.Tensor(obs), torch.Tensor(actions)).flatten()
            with torch.no_grad():
                targ_out = target_qf(torch.Tensor(next_obs), torch.Tensor(next_actions)).flatten()
                alpha = torch.exp(self.log_alpha)[0]
            bootstrapped_value = targ_out - (alpha * next_ll)
            bellman = torch.Tensor(rewards) + self.discount*(bootstrapped_value)
            q_objective = 0.5 * F.mse_loss(curr_q_val, bellman)
            qf_loss.append(q_objective.detach().numpy())
            qf_optimizer.zero_grad()
            q_objective.backward()
            qf_optimizer.step()
        tabular.record("qf_loss", np.mean(qf_loss))

    def optimize_policy(self, itr, samples):
        """ Optimize the policy based on the policy objective from the sac paper.

        Args:
            itr (int) - current training iteration
            samples() - samples recovered from the replay buffer
        Returns:
            None
        """

        obs = samples["observation"]
        # use the forward function instead of the get action function
        # in order to make sure that policy is differentiated. 
        action_dists = self.policy(torch.Tensor(obs))
        actions = action_dists.rsample()
        log_pi = action_dists.log_prob(actions)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi * self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            tabular.record("alpha_loss", alpha_loss.item())

        min_q = torch.min(self.qf1(torch.Tensor(obs), torch.Tensor(actions)), 
                            self.qf2(torch.Tensor(obs), torch.Tensor(actions)))
        with torch.no_grad():
            alpha = torch.exp(self.log_alpha)[0]
        policy_objective = ((alpha * log_pi) - min_q.flatten()).mean()
        logged_entropy = np.mean(alpha.detach().numpy() * -log_pi.detach().numpy())
        self.policy_optimizer.zero_grad()
        tabular.record("mean entropy", logged_entropy)
        policy_objective.backward()
        self.policy_optimizer.step()

    def _adjust_temperature(self, itr):        
        pass

    def update_targets(self):
        """Update parameters in the target q-functions."""
        # update for target_qf1
        target_qfs = [self.target_qf1, self.target_qf2]
        qfs = [self.qf1, self.qf2]
        for target_qf, qf in zip(target_qfs, qfs):
                for t_param, param in zip(target_qf.parameters(),
                                            qf.parameters()):
                        t_param.data.copy_(t_param.data * (1.0 - self.tau) +
                                        param.data * self.tau)
