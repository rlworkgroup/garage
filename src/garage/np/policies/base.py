"""Base class for policies based on numpy."""
import abc


class Policy(abc.ABC):
    """Base classe for policies based on numpy.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    """

    def __init__(self, env_spec):
        self._env_spec = env_spec

    @abc.abstractmethod
    def get_action(self, observation):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            (np.ndarray): Action sampled from the policy.

        """

    def reset(self, dones=None):
        """Reset the policy.

        If dones is None, it will be by default np.array([True]) which implies
        the policy will not be "vectorized", i.e. number of parallel
        environments for training data sampling = 1.

        Args:
            dones (numpy.ndarray): Bool that indicates terminal state(s).

        """

    @property
    def observation_space(self):
        """akro.Space: The observation space of the environment."""
        return self._env_spec.observation_space

    @property
    def action_space(self):
        """akro.Space: The action space for the environment."""
        return self._env_spec.action_space

    @property
    def recurrent(self):
        """Indicate whether the policy is recurrent.

        Returns:
            bool: True if policy is recurrent, False otherwise.

        """
        return False

    def log_diagnostics(self, paths):
        """Log extra information per iteration based on the collected paths.

        Args:
            paths (list[dict]): A list of collected paths

        """

    @property
    def state_info_keys(self):
        """Get keys describing policy's state.

        Returns:
            List[str]: keys for the information related to the policy's state
            when taking an action.

        """
        return list()

    def terminate(self):
        """Clean up operation."""


class StochasticPolicy(Policy):
    """Base class for stochastic policies implemented in numpy."""

    @property
    @abc.abstractmethod
    def distribution(self):
        """Get the distribution of the policy.

        Returns:
            garage.tf.distribution: The distribution of the policy.

        """

    @abc.abstractmethod
    def dist_info(self, obs, state_infos):
        """Return the distribution information about the actions.

        Args:
            obs (np.ndarray): observation values
            state_infos (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the observation

        """
