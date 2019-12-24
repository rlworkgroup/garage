"""Base class for policies based on numpy."""
import abc


class Policy(abc.ABC):
    """Base classe for policies based on numpy.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    Note:
        `policy_info` is passed as an argument in some APIs in Policy. This
        is intended to allow policies to pass state to themselves between
        calls. Users should consider its contents to be private, and should
        never assume this variable contains anything in particular, or even
        that its contents is consistent between calls to the same Policy
        object. In most use cases for non-recurrent policies, `policy_info`
        will be an empty dictionary while for recurrent-policies, it may
        store its previous hidden/cell state as the policy state.

        User should never mutate the contents of `policy_info` outside Policy,
        and no policy should ever require the user inspect or mutate
        `policy_info` as part of its interface.

        Implementers of Policy may occasionally find it useful to store some
        debugging information in `policy_info`, but any information
        implementers would like to be consistently logged by users should be
        exposed explicitly through the log_diagnostics interface.

    """

    def __init__(self, env_spec):
        self._env_spec = env_spec

    @abc.abstractmethod
    def get_action(self, observation, policy_info=None):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Observation from the environment.
            policy_info (dict): Info for the policy.

        Returns:
            np.ndarray: Action sampled from the policy.

        """

    @abc.abstractmethod
    def get_actions(self, observations, policy_infos=None):
        """Get actions sampled from the policy.

        Args:
            observations (list[np.ndarray]): Observations from the environment.
            policy_infos (dict): Infos for the policy.

        Returns:
            np.ndarray: Actions sampled from the policy.

        """

    def reset(self, policy_infos, dones=None):
        """Reset the policy.

        If dones is None, it will be by default np.array([True]) which implies
        the policy will not be "vectorized", i.e. number of parallel
        environments for training data sampling = 1.

        Args:
            policy_infos (dict): Infos for policy states.
            dones (numpy.ndarray): Bool that indicates terminal state(s).

        """

    def get_initial_state(self):
        """Initial state.

        Returns:
            dict: Initial state.

        """

    def get_initial_states(self, batch_size):
        """Initial states.

        Args:
            batch_size (int): Number of parallel environments for
                training data sampling.

        Returns:
            dict: Initial states.

        """

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
    def recurrent(self):
        """Indicate whether the policy is recurrent.

        Returns:
            bool: True if policy is recurrent, False otherwise.

        """
        return False

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
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec

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
