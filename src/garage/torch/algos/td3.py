"""TD3 model in Pytorch."""
import copy

from dowel import logger, tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import _Default, log_performance, make_optimizer
from garage._dtypes import TrajectoryBatch
from garage.misc import tensor_utils
from garage.np.algos import RLAlgorithm
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import (dict_np_to_torch, global_device, set_gpu_mode,
                          torch_to_np)



class TD3(RLAlgorithm):
    """Implementation of TD3.

    Based on https://arxiv.org/pdf/1802.09477.pdf.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy (actor network).
        qf1 (garage.torch.q_functions.QFunction): Q function (critic network).
        qf2 (garage.torch.q_functions.QFunction): Q function (critic network).
        replay_buffer (ReplayBuffer): Replay buffer.
        exploration_policy (garage.np.exploration_policies.ExplorationPolicy):
                Exploration strategy.
        target_update_tau (float): Interpolation parameter for doing the
            soft target update.
        discount (float): Discount factor (gamma) for the cumulative return.
        reward_scaling (float): Reward scaling.
        update_actor_interval (int): Policy (Actor network) update interval.
        max_action (float): Maximum action magnitude.
        max_episode_length (int): Maximum path length. The episode will
            terminate when length of trajectory reaches max_episode_length.
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
        num_evaluation_trajectories (int): The number of evaluation
            trajectories used for computing eval stats at the end of every
            epoch.

    """

    def __init__(
        self,
        env_spec,
        policy,
        qf1,
        qf2,
        replay_buffer,
        max_episode_length,
        grad_steps_per_env_step,
        exploration_policy=None,
        max_action=None,
        target_update_tau=0.005,
        discount=0.99,
        reward_scaling=1.,
        update_actor_interval=2,
        buffer_batch_size=64,
        min_buffer_size=1e4,
        exploration_noise=0.1,
        policy_noise=0.2,
        policy_noise_clip=0.5,
        clip_return=np.inf,
        policy_lr=_Default(1e-4),
        qf_lr=_Default(1e-3),
        policy_optimizer=torch.optim.Adam,
        qf_optimizer=torch.optim.Adam,
        num_evaluation_trajectories=10,
        steps_per_epoch=20,
        start_steps=1000,
    ):

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
        self._min_buffer_size = min_buffer_size
        self._buffer_batch_size = buffer_batch_size
        self._grad_steps_per_env_step = grad_steps_per_env_step
        self._update_actor_interval = update_actor_interval
        self._steps_per_epoch = steps_per_epoch
        self._start_steps = start_steps
        self._num_evaluation_trajectories = num_evaluation_trajectories
        self.max_episode_length = max_episode_length

        self._episode_policy_losses = []
        self._episode_qf_losses = []
        self._epoch_ys = []
        self._epoch_qs = []
        self._eval_env = None
        self.exploration_policy = exploration_policy
        self.worker_cls = FragmentWorker
        self.sampler_cls = LocalSampler

        self._replay_buffer = replay_buffer
        self.policy = policy
        self._qf_1 = qf1
        self._qf_2 = qf2
        self._target_policy = copy.deepcopy(self.policy)
        self._target_qf_1 = copy.deepcopy(self._qf_1)
        self._target_qf_2 = copy.deepcopy(self._qf_2)
        self._networks = [
            self.policy, self._qf_1, self._qf_2, self._target_policy,
            self._target_qf_1, self._target_qf_2
        ]

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

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        data = self.__dict__.copy()
        del data['_replay_buffer']
        del data['policy']
        del data['_qf_1']
        del data['_qf_2']
        del data['_target_policy']
        del data['_target_qf_1']
        del data['_target_qf_2']
        return data

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        self.__dict__.update(state)
        self._replay_buffer = self._replay_buffer
        self.policy = self.policy
        self._qf_1 = self._qf_1
        self._qf_2 = self._qf_2
        self._target_policy = self._target_policy
        self._target_qf_1 = self._target_qf_1
        self._target_qf_2 = self._target_qf_2

    def _get_action(self, action, noise_scale):
        """"Select action based on policy.
        
        Action can be added with noise.

        Args:
            action (float): Action.
            noise_scale (float): Noise scale added to action.

        Return:
            float: Action selected by the policy.
        """
        action += noise_scale * np.random.randn(self._action_dim)
        return np.clip(action, -self._max_action, self._max_action)



    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): Experiment runner, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.
        """
        if not self._eval_env:
            self._eval_env = runner.get_env_copy()
        last_returns = None
        runner.enable_logging = False

        for _ in runner.step_epochs():
            for cycle in range(self._steps_per_epoch):
                # Obtain trasnsition batch and store it in replay buffer
                runner.step_path = runner.obtain_trajectories(runner.step_itr)
                self._replay_buffer.add_trajectory_batch(runner.step_path)

                # Get action randomly from environment within warmup steps.
                # Afterwards, get action from policy.
                if runner.total_env_steps >= self._start_steps:
                    self._train_once(runner.step_itr)

                # Evaluate and log the results
                if (cycle == 0 and self._replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    runner.enable_logging = True
                    eval_eps = self._evaluate_policy()
                    log_performance(runner.step_path,
                                    eval_eps,
                                    discount=self._discount,
                                    prefix='training')
                    last_returns = log_performance(runner.step_itr,
                                                   eval_eps,
                                                   discount=self._discount,
                                                   prefix='evaluation')
                runner.step_itr += 1

        return np.mean(last_returns)

    def _train_once(self, itr):
        """Perform one iteration of training.

        Args:
            itr (int): Iteration number.

        """
        for _ in range(self._grad_steps_per_env_step):
            if (self._replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                # Sample from buffer
                samples = self._replay_buffer.sample_transitions(
                    self._buffer_batch_size)
                samples = dict_np_to_torch(samples)

                # Optimize
                qf_loss, y, q, policy_loss = torch_to_np(
                    self._optimize_policy(samples, itr))

                self._episode_policy_losses.append(policy_loss)
                self._episode_qf_losses.append(qf_loss)
                self._epoch_ys.append(y)
                self._epoch_qs.append(q)

        if itr % self._steps_per_epoch == 0:
            logger.log('Training finished')
            epoch = itr / self._steps_per_epoch

            if (self._replay_buffer.n_transitions_stored >=
                    self._min_buffer_size):
                tabular.record('Epoch', epoch)
                self._log_statistics()

    # pylint: disable=invalid-unary-operand-type
    def _optimize_policy(self, samples_data, itr):
        """Perform algorithm optimization.

        Args:
            samples_data (dict): Processed batch data.
            itr (int): Iteration count.

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
        rewards = samples_data['rewards'].reshape(-1, 1)
        terminals = samples_data['terminals'].reshape(-1, 1)
        actions = samples_data['actions']
        observations = samples_data['observations']
        next_observations = samples_data['next_observations']

        next_inputs = next_observations
        inputs = observations
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions)* self._policy_noise).clamp(
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
        self._qf_optimizer_1.zero_grad()
        self._qf_optimizer_2.zero_grad()
        critic_loss.backward()
        self._qf_optimizer_1.step()
        self._qf_optimizer_2.step()

        # Deplay policy updates
        if itr % self._update_actor_interval == 0:
            # Compute actor loss
            actions = self.policy(inputs)
            self._actor_loss = -self._qf_1(inputs, actions).mean()

            # Optimize actor
            self._policy_optimizer.zero_grad()
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
        paths = []

        for _ in range(self._num_evaluation_trajectories):
            observations, actions, rewards, dones, agent_infos, env_infos = \
                [], [], [], [], [], []
            obs, path_length, episode_reward, agenet_info = \
                self._eval_env.reset(), 0, 0, dict()
            self.policy.reset()

            while path_length < (self.max_episode_length or np.inf):
                obs = self._eval_env.observation_space.flatten(obs)

                # select action according to policy
                with torch.no_grad():
                    a = self.policy(torch.Tensor(obs).unsqueeze(0))
                    action = self._get_action(a, self._exploration_noise)
                    action = action.squeeze(0).numpy()

                # Perform action
                next_obs, reward, done, env_info = self._eval_env.step(action)
                path_length += 1
                episode_reward += reward

                observations.append(obs)
                rewards.append(episode_reward)
                actions.append(action)
                agent_infos.append(agenet_info)
                env_infos.append(env_info)
                dones.append(done)

                if done:
                    break

                # Update state
                obs = next_obs

            path = dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
                env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
                dones=np.array(dones),
            )
            paths.append(path)
        return TrajectoryBatch.from_trajectory_list(self._eval_env.spec, paths)

    def _update_network_parameters(self):
        """Update parameters in actor network and critic networks."""
        for target_param, param in zip(self._target_qf_1.parameters(),
                                       self._qf_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self._tau) +
                                    param.data * self._tau)

        for target_param, param in zip(self._target_qf_2.parameters(),
                                       self._qf_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self._tau) +
                                    param.data * self._tau)

        for target_param, param in zip(self._target_policy.parameters(),
                                       self.policy.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self._tau) +
                                    param.data * self._tau)

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

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if torch.cuda.is_available():
            set_gpu_mode(True)
        else:
            set_gpu_mode(False)

        if device is None:
            device = global_device()
        for net in self._networks:
            net.to(device)
