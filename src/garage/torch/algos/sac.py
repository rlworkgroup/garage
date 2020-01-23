"""This modules creates a sac model in PyTorch."""
import copy

from dowel import logger, tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage.np.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from garage.torch.utils import np_to_torch, torch_to_np
from collections import deque

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
                 gradient_steps_per_itr,
                 alpha=None,
                 target_entropy=None,
                 initial_log_entropy=0.,
                 use_automatic_entropy_tuning=True,
                 discount=0.99,
                 max_path_length=None,
                 buffer_batch_size=64,
                 min_buffer_size=int(1e4),
                 target_update_tau=5e-3,
                 policy_lr=3e-4,
                 qf_lr=3e-4,
                 reward_scale=1.0,
                 optimizer=torch.optim.Adam,
                 smooth_return=True,
                 input_include_goal=False):

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.replay_buffer = replay_buffer
        self.tau = target_update_tau
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.initial_log_entropy = initial_log_entropy
        self.gradient_steps = gradient_steps_per_itr
        self.evaluate = False
        self.input_include_goal = input_include_goal

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         qf=qf1,
                         n_train_steps=self.gradient_steps,
                         max_path_length=max_path_length,
                         buffer_batch_size=buffer_batch_size,
                         min_buffer_size=min_buffer_size,
                         replay_buffer=replay_buffer,
                         use_target=True,
                         discount=discount,
                         smooth_return=smooth_return)
        self.reward_scale = reward_scale
        # use 2 target q networks
        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)
        self.policy_optimizer = optimizer(self.policy.parameters(),
                                          lr=self.policy_lr)
        self.qf1_optimizer = optimizer(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optimizer(self.qf2.parameters(), lr=self.qf_lr)
        # automatic entropy coefficient tuning
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning and not alpha:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            self.log_alpha = torch.tensor([self.initial_log_entropy], dtype=torch.float, requires_grad=True)
            self.alpha_optimizer = optimizer([self.log_alpha], lr=self.policy_lr)
        else:
            self.alpha = alpha

        self.episode_rewards = deque(maxlen=30)
        self.success_history = []

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            The average return in last epoch cycle.

        """
        def train_helper(runner, batch_size):
            runner.step_path = runner.obtain_samples(runner.step_itr, batch_size)
            for sample in runner.step_path:
                self.replay_buffer.store(obs=sample[1],
                                         act=sample[2],
                                         rew=sample[3],
                                         next_obs=sample[4],
                                         done=sample[5])
            tabular.record("buffer_size", self.replay_buffer.n_transitions_stored)
            self.episode_rewards.append(sum([sample[3] for sample in runner.step_path]))
            tabular.record('average_return', np.mean(self.episode_rewards))
            last_return = self.train_once(runner.step_itr,
                                            runner.step_path)
            
            runner.step_itr += 1
            return last_return

        last_return = None

        for epoch in runner.step_epochs():
            if self.replay_buffer.n_transitions_stored < self.min_buffer_size:
                batch_size = self.min_buffer_size
            else:
                batch_size = None
            train_helper(runner, batch_size)
        return last_return

    def train_once(self, itr, paths):
        """
        """
        if self.replay_buffer.n_transitions_stored >= self.min_buffer_size:  # noqa: E501
            for gradient_step in range(self.gradient_steps):
                    samples = self.replay_buffer.sample(self.buffer_batch_size)
                    self.optimize_policy(itr, gradient_step, samples)
                    self.update_targets()

        return np.mean(self.episode_rewards)

    def temperature_objective(self, log_pi):
        """
        implemented inside optimize_policy
        """
        alpha_loss = 0
        if self.use_automatic_entropy_tuning:
            alpha_loss = (-(self.log_alpha) * (log_pi.detach() + self.target_entropy)).mean()
        return alpha_loss

    def actor_objective(self, obs, log_pi, new_actions):
        alpha = self.log_alpha.detach().exp()
        min_q_new_actions = torch.min(self.qf1(torch.Tensor(obs), torch.Tensor(new_actions)), 
                            self.qf2(torch.Tensor(obs), torch.Tensor(new_actions)))
        policy_objective = ((alpha * log_pi) - min_q_new_actions.flatten()).mean()
        return policy_objective

    def critic_objective(self,samples):
        '''
        QF Loss
        '''
        obs = samples["observation"]
        actions = samples["action"]
        rewards = samples["reward"]
        terminals = samples["terminal"]
        next_obs = samples["next_observation"]

        alpha = self.log_alpha.detach().exp()

        q1_pred = self.qf1(torch.Tensor(obs), torch.Tensor(actions))
        q2_pred = self.qf2(torch.Tensor(obs), torch.Tensor(actions))

        new_next_actions_dist = self.policy(torch.Tensor(next_obs))
        new_next_actions_pre_tanh, new_next_actions = new_next_actions_dist.rsample_return_pre_tanh_value()
        new_log_pi = new_next_actions_dist.log_prob(value=new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)

        target_q_values = torch.min(
            self.target_qf1(torch.Tensor(next_obs), new_next_actions),
            self.target_qf2(torch.Tensor(next_obs), new_next_actions)
        ).flatten() - (alpha * new_log_pi)
        with torch.no_grad():
            q_target = self.reward_scale * torch.Tensor(rewards) + (1. - torch.Tensor(terminals)) * self.discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        return qf1_loss, qf2_loss

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

    def optimize_policy(self, itr, gradient_step, samples):
        """ Optimize the policy based on the policy objective from the sac paper.

        Args:
            itr (int) - current training iteration
            samples() - samples recovered from the replay buffer
        Returns:
            None
        """

        obs = samples["observation"]

        # ================update q functions===============#
        qf1_loss, qf2_loss = self.critic_objective(samples)
        # if(self.gradient_steps == 1):
        #     tabular.record("qf_loss/{}".format("qf1_loss"), float(qf1_loss))
        #     tabular.record("qf_loss/{}".format("qf2_loss"), float(qf2_loss))
        # elif ((gradient_step % (self.gradient_steps - 1) == 0) and gradient_step != 0):
        tabular.record("qf_loss/{}".format("qf1_loss"), float(qf1_loss))
        tabular.record("qf_loss/{}".format("qf2_loss"), float(qf2_loss))
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        # for name, param in self.qf1.named_parameters():
        #     if param.requires_grad and (param.grad is not None):
        #         tabular.record("qf1_grads/{}".format(name), float(param.grad.norm()))
        #         tabular.record("qf2_grads_max/{}".format(name), float(param.grad.max()))
        #         tabular.record("qf2_grads_min/{}".format(name), float(param.grad.min()))
        self.qf1_optimizer.step()
        
        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        # for name, param in self.qf2.named_parameters():
        #     if param.requires_grad and (param.grad is not None):
        #         tabular.record("qf2_grads/{}".format(name), float(param.grad.norm()))
        #         tabular.record("qf2_grads_max/{}".format(name), float(param.grad.max()))
        #         tabular.record("qf2_grads_min/{}".format(name), float(param.grad.min()))
        self.qf2_optimizer.step()
        #===================================================#

        #===================update policy===================#
        action_dists = self.policy(torch.Tensor(obs))
        # log evaluation statistics
        if(self.gradient_steps == 1):
            entropy = action_dists.entropy()
            tabular.record("policy/mu_entropy", torch.mean(entropy).detach().numpy().item())
            tabular.record("policy/mu_stdev", torch.mean(action_dists.variance.detach().flatten()).item()**0.5)
            tabular.record("policy/min_stdev", torch.min(action_dists.variance.detach().flatten()).item()**0.5)
            tabular.record("policy/max_stdev", torch.max(action_dists.variance.detach().flatten()).item()**0.5)
            tabular.record("policy/mu_mean", torch.mean(action_dists.mean.detach()).item())
        elif ((gradient_step % (self.gradient_steps - 1) == 0) and gradient_step != 0):
            entropy = action_dists.entropy()
            tabular.record("policy/mu_entropy", torch.mean(entropy).detach().numpy().item())
            tabular.record("policy/mu_stdev", torch.mean(action_dists.variance.detach().flatten()).item()**0.5)
            tabular.record("policy/min_stdev", torch.min(action_dists.variance.detach().flatten()).item()**0.5)
            tabular.record("policy/max_stdev", torch.max(action_dists.variance.detach().flatten()).item()**0.5)
            tabular.record("policy/mu_mean", torch.mean(action_dists.mean.detach()).item())
        new_actions_pre_tanh, new_actions = action_dists.rsample_return_pre_tanh_value()
        log_pi = action_dists.log_prob(value=new_actions, pre_tanh_value=new_actions_pre_tanh)

        policy_loss = self.actor_objective(obs, log_pi, new_actions)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # for name, param in self.policy.named_parameters():
        #     if param.requires_grad and (param.grad is not None):
        #         tabular.record("policy_grads/{}".format(name), float(param.grad.norm()))
        #         tabular.record("policy_grads_max/{}".format(name), float(param.grad.max()))
        #         tabular.record("policy_grads_min/{}".format(name), float(param.grad.min()))

        self.policy_optimizer.step()
        if(self.gradient_steps == 1):
            tabular.record("policy_loss", policy_loss.item())
        elif ((gradient_step % (self.gradient_steps - 1) == 0) and gradient_step != 0):
            tabular.record("policy_loss", policy_loss.item())
        #===================================================#

        #===================Update alpha====================#
        if self.use_automatic_entropy_tuning:
            alpha_loss = (-(self.log_alpha)* (log_pi.detach() + self.target_entropy)).mean()
            alpha_loss = self.temperature_objective(log_pi)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            if(self.gradient_steps == 1):
                tabular.record("alpha", torch.exp(self.log_alpha.detach()).item())
            elif ((gradient_step % (self.gradient_steps - 1) == 0) and gradient_step != 0):
                tabular.record("alpha", torch.exp(self.log_alpha.detach()).item())
