"""Base class for policies based on numpy."""
import abc


class Policy(abc.ABC):
    """Base class for policies based on numpy."""

    @abc.abstractmethod
    def get_action(self, observation):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            Tuple[np.ndarray, dict[str,np.ndarray]]: Actions and extra agent
                infos.

        """

    @abc.abstractmethod
    def get_actions(self, observations):
        """Get actions given observations.

        Args:
            observations (torch.Tensor): Observations from the environment.

        Returns:
            Tuple[np.ndarray, dict[str,np.ndarray]]: Actions and extra agent
                infos.

        """

    def reset(self, do_resets=None):
        """Reset the policy.

        This is effective only to recurrent policies.

        do_resets is an array of boolean indicating
        which internal states to be reset. The length of do_resets should be
        equal to the length of inputs, i.e. batch size.

        Args:
            do_resets (numpy.ndarray): Bool array indicating which states
                to be reset.

        """

    @property
    def name(self):
        """Name of policy.

        Returns:
            str: Name of policy

        """

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """

    @property
    def observation_space(self):
        """Observation space.

        Returns:
            akro.Space: The observation space of the environment.

        """
        return self.env_spec.observation_space

    @property
    def action_space(self):
        """Action space.

        Returns:
            akro.Space: The action space of the environment.

        """
        return self.env_spec.action_space
