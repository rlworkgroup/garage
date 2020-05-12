"""Base class for policies in TensorFlow."""
import abc

from garage.tf.models import Module, StochasticModule


class Policy(Module):
    """Base class for policies in TensorFlow.

    Args:
        name (str): Policy name, also the variable scope.
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    """

    def __init__(self, name, env_spec):
        super().__init__(name)
        self._env_spec = env_spec

    @abc.abstractmethod
    def get_action(self, observation):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            (np.ndarray): Action sampled from the policy.

        """

    @abc.abstractmethod
    def get_actions(self, observations):
        """Get action sampled from the policy.

        Args:
            observations (list[np.ndarray]): Observations from the environment.

        Returns:
            (np.ndarray): Actions sampled from the policy.

        """

    @property
    def vectorized(self):
        """Boolean for vectorized.

        Returns:
            bool: Indicates whether the policy is vectorized. If True, it
                should implement get_actions(), and support resetting with
                multiple simultaneous states.

        """
        return False

    @property
    def observation_space(self):
        """Observation space.

        Returns:
            akro.Space: The observation space of the environment.

        """
        return self._env_spec.observation_space

    @property
    def action_space(self):
        """Action space.

        Returns:
            akro.Space: The action space of the environment.

        """
        return self._env_spec.action_space

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec

    def log_diagnostics(self, paths):
        """Log extra information per iteration based on the collected paths.

        Args:
            paths (dict[numpy.ndarray]): Sample paths.

        """


# pylint: disable=abstract-method
class StochasticPolicy(Policy, StochasticModule):
    """Stochastic Policy."""
