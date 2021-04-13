"""TD3 model in Pytorch."""
import copy

from dowel import logger, tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import (_Default, log_performance, make_optimizer,
                    obtain_evaluation_episodes)
from garage.np.algos import RLAlgorithm
from garage.torch import (as_torch_dict, global_device, soft_update_model,
                          torch_to_np)
from garage.torch._functions import zero_optim_grads


class TD3(RLAlgorithm):
    """Implementation of TD3.

    Based on https://arxiv.org/pdf/1802.09477.pdf.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy (actor network).
        qf1 (garage.torch.q_functions.QFunction): Q function (critic network).
        qf2 (garage.torch.q_functions.QFunction): Q function (critic network).
        replay_buffer (ReplayBuffer): Replay buffer.
        sampler (garage.sampler.Sampler): Sampler.
        replay_buffer_size (int): Size of the replay buffer
        exploration_policy (garage.np.exploration_policies.ExplorationPolicy):
                Exploration strategy.
        uniform_random_policy
                (garage.np.exploration_policies.ExplorationPolicy):
                Uniform random exploration strategy.
        target_update_tau (float): Interpolation parameter for doing the
            soft target update.
        discount (float): Discount factor (gamma) for the cumulative return.
        reward_scaling (float): Reward scaling.
        update_actor_interval (int): Policy (Actor network) update interval.
        max_action (float): Maximum action magnitude.
        buffer_batch_size (int): Size of replay buffer.
        min_buffer_size (int): The minimum buffer size for replay buffer.
        policy_noise (float): Policy (actor) noise.
        policy_noise_clip (float): Noise clip.
        exploration_noise (float): Exploration noise.
        clip_return (float): Clip return to be in [-clip_return,
            clip_return].
        policy_lr (float): Learning rate for training policy network.
        qf_lr (float): Learning rate for training Q network.
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
        steps_per_epoch (int): Number of train_once calls per epoch.
        grad_steps_per_env_step (int): Number of gradient steps taken per
            environment step sampled.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If None, defaults to
            `env_spec.max_episode_length`.
        num_evaluation_episodes (int): The number of evaluation
            trajectories used for computing eval stats at the end of every
            epoch.
        start_steps (int): The number of steps for warming up before
             selecting actions according to policy.
        update_after (int): The number of steps to perform before policy
            is updated.
        use_deterministic_evaluation (bool): True if the trained policy
            should be evaluated deterministically.

    """

    def __init__(
            self,
            env_spec,
            policy,
            qf1,
            qf2,
            replay_buffer,
            sampler,
            *,  # Everything after this is numbers.
            max_episode_length_eval=None,
            grad_steps_per_env_step,
            exploration_policy,
            uniform_random_policy=None,
            max_action=None,
            target_update_tau=0.005,
            discount=0.99,
            reward_scaling=1.,
            update_actor_interval=2,
            buffer_batch_size=64,
            replay_buffer_size=1e6,
            min_buffer_size=1e4,
            exploration_noise=0.1,
            policy_noise=0.2,
            policy_noise_clip=0.5,
            clip_return=np.inf,
            policy_lr=_Default(1e-4),
            qf_lr=_Default(1e-3),
            policy_optimizer=torch.optim.Adam,
            qf_optimizer=torch.optim.Adam,
            num_evaluation_episodes=10,
            steps_per_epoch=20,
            start_steps=10000,
            update_after=1000,
            use_deterministic_evaluation=False):

        self._env_spec = env_spec
        action_bound = self._env_spec.action_space.high[0]
        self._max_action = action_bound if max_action is None else max_action
        self._action_dim = self._env_spec.action_space.shape[0]
        self._tau = target_update_tau
        self._discount = discount
        self._reward_scaling = reward_scaling
        self._exploration_noise = exploration_noise
        self._policy_noise = policy_noise
        self._policy_noise_clip = policy_noise_clip
        self._clip_return = clip_return
        self._replay_buffer_size = replay_buffer_size
        self._min_buffer_size = min_buffer_size
        self._buffer_batch_size = buffer_batch_size
        self._grad_steps_per_env_step = grad_steps_per_env_step
        self._update_actor_interval = update_actor_interval
        self._steps_per_epoch = steps_per_epoch
        self._start_steps = start_steps
        self._update_after = update_after
        self._num_evaluation_episodes = num_evaluation_episodes
        self.max_episode_length = env_spec.max_episode_length
        self._max_episode_length_eval = env_spec.max_episode_length

        if max_episode_length_eval is not None:
            self._max_episode_length_eval = max_episode_length_eval
        self._use_deterministic_evaluation = use_deterministic_evaluation

        self._episode_policy_losses = []
        self._episode_qf_losses = []
        self._epoch_ys = []
        self._epoch_qs = []
        self._eval_env = None
        self.exploration_policy = exploration_policy
        self._uniform_random_policy = uniform_random_policy
        self._sampler = sampler

        self._replay_buffer = replay_buffer
        self.policy = policy
        self._qf_1 = qf1
        self._qf_2 = qf2
        self._target_policy = copy.deepcopy(self.policy)
        self._target_qf_1 = copy.deepcopy(self._qf_1)
        self._target_qf_2 = copy.deepcopy(self._qf_2)

        self._policy_optimizer = make_optimizer(policy_optimizer,
                                                module=self.policy,
                                                lr=policy_lr)
        self._qf_optimizer_1 = make_optimizer(qf_optimizer,
                                              module=self._qf_1,
                                              lr=qf_lr)
        self._qf_optimizer_2 = make_optimizer(qf_optimizer,
                                              module=self._qf_2,
                                              lr=qf_lr)
        self._actor_loss = torch.zeros(1)

    def _get_action(self, action, noise_scale):
        """Select action based on policy.

        Action can be added with noise.

        Args:
            action (float): Action.
            noise_scale (float): Noise scale added to action.

        Return:
            float: Action selected by the policy.
        """
        action += noise_scale * np.random.randn(self._action_dim)
        # pylint: disable=invalid-unary-operand-type
        return np.clip(action, -self._max_action, self._max_action)

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Experiment trainer, which provides services
                such as snapshotting and sampler control.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        trainer.enable_logging = False
        for _ in trainer.step_epochs():
            for cycle in range(self._steps_per_epoch):
                # Obtain trasnsition batch and store it in replay buffer.
                # Get action randomly from environment within warm-up steps.
                # Afterwards, get action from policy.
                if self._uniform_random_policy and \
                        trainer.step_itr < self._start_steps:
                    trainer.step_episode = trainer.obtain_episodes(
                        trainer.step_itr,
                        agent_update=self._uniform_random_policy)
                else:
                    trainer.step_episode = trainer.obtain_episodes(
                        trainer.step_itr, agent_update=self.exploration_policy)
                self._replay_buffer.add_episode_batch(trainer.step_episode)

                # Update after warm-up steps.
                if trainer.total_env_steps >= self._update_after:
                    self._train_once(trainer.step_itr)

                # Evaluate and log the results.
                if (cycle == 0 and self._replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    trainer.enable_logging = True
                    eval_eps = self._evaluate_policy()
                    log_performance(trainer.step_episode,
                                    eval_eps,
                                    discount=self._discount,
                                    prefix='Training')
                    log_performance(trainer.step_itr,
                                    eval_eps,
                                    discount=self._discount,
                                    prefix='Evaluation')
                trainer.step_itr += 1

    def _train_once(self, itr):
        """Perform one iteration of training.

        Args:
            itr (int): Iteration number.

        """
        for grad_step_timer in range(self._grad_steps_per_env_step):
            if (self._replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                # Sample from buffer
                samples = self._replay_buffer.sample_transitions(
                    self._buffer_batch_size)
                samples = as_torch_dict(samples)

                # Optimize
                qf_loss, y, q, policy_loss = torch_to_np(
                    self._optimize_policy(samples, grad_step_timer))

                self._episode_policy_losses.append(policy_loss)
                self._episode_qf_losses.append(qf_loss)
                self._epoch_ys.append(y)
                self._epoch_qs.append(q)

        if itr % self._steps_per_epoch == 0:
            logger.log('Training finished')
            epoch = itr // self._steps_per_epoch

            if (self._replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                tabular.record('Epoch', epoch)
                self._log_statistics()

    # pylint: disable=invalid-unary-operand-type
    def _optimize_policy(self, samples_data, grad_step_timer):
        """Perform algorithm optimization.

        Args:
            samples_data (dict): Processed batch data.
            grad_step_timer (int): Iteration number of the gradient time
                taken in the env.

        Returns:
            float: Loss predicted by the q networks
                (critic networks).
            float: Q value (min) predicted by one of the
                target q networks.
            float: Q value (min) predicted by one of the
                current q networks.
            float: Loss predicted by the policy
                (action network).

        """
        rewards = samples_data['rewards'].to(global_device()).reshape(-1, 1)
        terminals = samples_data['terminals'].to(global_device()).reshape(
            -1, 1)
        actions = samples_data['actions'].to(global_device())
        observations = samples_data['observations'].to(global_device())
        next_observations = samples_data['next_observations'].to(
            global_device())

        next_inputs = next_observations
        inputs = observations
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(
                -self._policy_noise_clip, self._policy_noise_clip)
            next_actions = (self._target_policy(next_inputs) + noise).clamp(
                -self._max_action, self._max_action)

            # Compute the target Q value
            target_Q1 = self._target_qf_1(next_inputs, next_actions)
            target_Q2 = self._target_qf_2(next_inputs, next_actions)
            target_q = torch.min(target_Q1, target_Q2)
            target_Q = rewards * self._reward_scaling + (
                1. - terminals) * self._discount * target_q

        # Get current Q values
        current_Q1 = self._qf_1(inputs, actions)
        current_Q2 = self._qf_2(inputs, actions)
        current_Q = torch.min(current_Q1, current_Q2)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        # Optimize critic
        zero_optim_grads(self._qf_optimizer_1)
        zero_optim_grads(self._qf_optimizer_2)
        critic_loss.backward()
        self._qf_optimizer_1.step()
        self._qf_optimizer_2.step()

        # Deplay policy updates
        if grad_step_timer % self._update_actor_interval == 0:
            # Compute actor loss
            actions = self.policy(inputs)
            self._actor_loss = -self._qf_1(inputs, actions).mean()

            # Optimize actor
            zero_optim_grads(self._policy_optimizer)
            self._actor_loss.backward()
            self._policy_optimizer.step()

            # update target networks
            self._update_network_parameters()

        return (critic_loss.detach(), target_Q, current_Q.detach(),
                self._actor_loss.detach())

    def _evaluate_policy(self):
        """Evaluate the performance of the policy via deterministic rollouts.

        Statistics such as (average) discounted return and success rate are
            recorded.

        Returns:
            TrajectoryBatch: Evaluation trajectories, representing the best
                current performance of the algorithm.

        """
        return obtain_evaluation_episodes(
            self.exploration_policy,
            self._eval_env,
            self._max_episode_length_eval,
            num_eps=self._num_evaluation_episodes,
            deterministic=self._use_deterministic_evaluation)

    def _update_network_parameters(self):
        """Update parameters in actor network and critic networks."""
        soft_update_model(self._target_qf_1, self._qf_1, self._tau)
        soft_update_model(self._target_qf_2, self._qf_2, self._tau)
        soft_update_model(self._target_policy, self.policy, self._tau)

    def _log_statistics(self):
        """Output training statistics to dowel such as losses and returns."""
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

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy, self._qf_1, self._qf_2, self._target_policy,
            self._target_qf_1, self._target_qf_2
        ]

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        device = device or global_device()
        for net in self.networks:
            net.to(device)
