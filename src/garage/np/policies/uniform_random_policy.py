"""Uniform random exploration strategy."""
import gym

from garage.np.policies.policy import Policy


class UniformRandomPolicy(Policy):
    """Action taken is uniformly random.

    Args:
        env_spec (EnvSpec): Environment spec to explore.

    """

    def __init__(
        self,
        env_spec,
    ):
        assert isinstance(env_spec.action_space, gym.spaces.Box)
        assert len(env_spec.action_space.shape) == 1
        self._env_spec = env_spec
        self._action_space = env_spec.action_space
        self._iteration = 0

    def reset(self, do_resets=None):
        """Reset the state of the exploration.

        Args:
            do_resets (List[bool] or numpy.ndarray or None): Which
                vectorization states to reset.

        """
        self._iteration += 1
        super().reset(do_resets)

    def get_action(self, observation):
        """Get action from this policy for the input observation.

        Args:
            observation(numpy.ndarray): Observation from the environment.

        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).

        """
        return self._env_spec.action_space.sample(), dict()

    def get_actions(self, observations):
        """Get actions from this policy for the input observation.

        Args:
            observations(list): Observations from the environment.

        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).

        """
        return [self._env_spec.action_space.sample()
                for obs in observations], dict()
