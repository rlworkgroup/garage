"""Base Policy."""
import abc


class Policy(abc.ABC):
    """
    Policy base class without Parameterzied.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
    """

    def __init__(self, env_spec):
        self._env_spec = env_spec

    @abc.abstractmethod
    def get_action(self, observation):
        """Get action given observation."""
        pass

    @abc.abstractmethod
    def get_actions(self, observations):
        """Get actions given observations."""
        pass

    @property
    def observation_space(self):
        """Observation space."""
        return self._env_spec.observation_space

    @property
    def action_space(self):
        """Policy action space."""
        return self._env_spec.action_space

    @property
    def vectorized(self):
        """Vectorized or not."""
        return False
