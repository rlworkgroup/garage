"""Gaussian exploration strategy."""
import akro
from dowel import tabular
import numpy as np

from garage.np.exploration_policies.exploration_policy import ExplorationPolicy


class AddGaussianNoise(ExplorationPolicy):
    """Add Gaussian noise to the action taken by the deterministic policy.

    Args:
        env_spec (EnvSpec): Environment spec to explore.
        policy (garage.Policy): Policy to wrap.
        total_timesteps (int): Total steps in the training, equivalent to
            max_episode_length * n_epochs.
        max_sigma (float): Action noise standard deviation at the start of
            exploration.
        min_sigma (float): Action noise standard deviation at the end of the
            decay period.
        decay_ratio (float): Fraction of total steps for epsilon decay.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 total_timesteps,
                 max_sigma=1.0,
                 min_sigma=0.1,
                 decay_ratio=1.0):
        assert isinstance(env_spec.action_space, akro.Box)
        assert len(env_spec.action_space.shape) == 1
        super().__init__(policy)
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = int(total_timesteps * decay_ratio)
        self._action_space = env_spec.action_space
        self._decrement = (self._max_sigma -
                           self._min_sigma) / self._decay_period
        self._total_env_steps = 0
        self._last_total_env_steps = 0

    def get_action(self, observation):
        """Get action from this policy for the input observation.

        Args:
            observation(numpy.ndarray): Observation from the environment.

        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).

        """
        action, agent_info = self.policy.get_action(observation)
        action = np.clip(
            action + np.random.normal(size=action.shape) * self._sigma(),
            self._action_space.low, self._action_space.high)
        self._total_env_steps += 1
        return action, agent_info

    def get_actions(self, observations):
        """Get actions from this policy for the input observation.

        Args:
            observations(list): Observations from the environment.

        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).

        """
        actions, agent_infos = self.policy.get_actions(observations)
        for itr, _ in enumerate(actions):
            actions[itr] = np.clip(
                actions[itr] +
                np.random.normal(size=actions[itr].shape) * self._sigma(),
                self._action_space.low, self._action_space.high)
            self._total_env_steps += 1
        return actions, agent_infos

    def _sigma(self):
        """Get the current sigma.

        Returns:
            double: Sigma.

        """
        if self._total_env_steps >= self._decay_period:
            return self._min_sigma
        return self._max_sigma - self._decrement * self._total_env_steps

    def update(self, episode_batch):
        """Update the exploration policy using a batch of trajectories.

        Args:
            episode_batch (EpisodeBatch): A batch of trajectories which
                were sampled with this policy active.

        """
        self._total_env_steps = (self._last_total_env_steps +
                                 np.sum(episode_batch.lengths))
        self._last_total_env_steps = self._total_env_steps
        tabular.record('AddGaussianNoise/Sigma', self._sigma())

    def get_param_values(self):
        """Get parameter values.

        Returns:
            list or dict: Values of each parameter.

        """
        return {
            'total_env_steps': self._total_env_steps,
            'inner_params': self.policy.get_param_values()
        }

    def set_param_values(self, params):
        """Set param values.

        Args:
            params (np.ndarray): A numpy array of parameter values.

        """
        self._total_env_steps = params['total_env_steps']
        self.policy.set_param_values(params['inner_params'])
        self._last_total_env_steps = self._total_env_steps
