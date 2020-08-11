"""Gaussian exploration strategy."""
import gym
import numpy as np

from garage.np.exploration_policies.exploration_policy import ExplorationPolicy


class AddGaussianNoise(ExplorationPolicy):
    """Add Gaussian noise to the action taken by the deterministic policy.

    Args:
        env_spec (EnvSpec): Environment spec to explore.
        policy (garage.Policy): Policy to wrap.
        max_sigma (float): Action noise standard deviation at the start of
            exploration.
        min_sigma (float): Action noise standard deviation at the end of the
            decay period.
        decay_period (int): Number of episodes over which to linearly decay
            sigma from max_sigma to min_sigma.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 max_sigma=1.0,
                 min_sigma=0.1,
                 decay_period=1000000):
        assert isinstance(env_spec.action_space, gym.spaces.Box)
        assert len(env_spec.action_space.shape) == 1
        super().__init__(policy)
        self._max_sigma = max_sigma
        self._min_sigma = min_sigma
        self._decay_period = decay_period
        self._action_space = env_spec.action_space
        self._iteration = 0

    def reset(self, dones=None):
        """Reset the state of the exploration.

        Args:
            dones (List[bool] or numpy.ndarray or None): Which vectorization
                states to reset.

        """
        self._iteration += 1
        super().reset(dones)

    def get_action(self, observation):
        """Get action from this policy for the input observation.

        Args:
            observation(numpy.ndarray): Observation from the environment.

        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).

        """
        action, agent_info = self.policy.get_action(observation)
        sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(
            1.0, self._iteration * 1.0 / self._decay_period)
        return np.clip(action + np.random.normal(size=action.shape) * sigma,
                       self._action_space.low,
                       self._action_space.high), agent_info

    def get_actions(self, observations):
        """Get actions from this policy for the input observation.

        Args:
            observations(list): Observations from the environment.

        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).

        """
        actions, agent_infos = self.policy.get_actions(observations)
        sigma = self._max_sigma - (self._max_sigma - self._min_sigma) * min(
            1.0, self._iteration * 1.0 / self._decay_period)
        return np.clip(actions + np.random.normal(size=actions.shape) * sigma,
                       self._action_space.low,
                       self._action_space.high), agent_infos
