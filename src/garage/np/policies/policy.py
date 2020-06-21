"""Base class for policies based on numpy."""
import abc


class Policy(abc.ABC):
    """Base class for policies based on numpy."""

    @abc.abstractmethod
    def get_action(self, observation):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Observation from the environment.

        """

    @abc.abstractmethod
    def get_actions(self, observations):
        """Get actions given observations.

        Args:
            observations (torch.Tensor): Observations from the environment.

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
    def observation_space(self):
        """akro.Space: The observation space of the environment."""

    @property
    def action_space(self):
        """akro.Space: The action space for the environment."""

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """

    @property
    def name(self):
        """Name of policy.

        Returns:
            str: Name of policy

        """

    def log_diagnostics(self, paths):
        """Log extra information per iteration based on the collected paths.

        Args:
            paths (list[dict]): A list of collected paths

        """

    @property
    def state_info_specs(self):
        """State info specification.

        Returns:
            List[str]: keys and shapes for the information related to the
                module's state when taking an action.

        """
        return list()

    @property
    def state_info_keys(self):
        """State info keys.

        Returns:
            List[str]: keys for the information related to the module's state
                when taking an input.

        """
        return [k for k, _ in self.state_info_specs]
