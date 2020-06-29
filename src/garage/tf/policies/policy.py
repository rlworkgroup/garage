"""Base class for policies in TensorFlow."""
import abc

from garage.np.policies import Policy as BasePolicy


class Policy(BasePolicy):
    """Base class for policies in TensorFlow."""

    @abc.abstractmethod
    def get_action(self, observation):
        """Get action sampled from the policy.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            Tuple[np.ndarray, dict[str,np.ndarray]]: Action and extra agent
                info.

        """

    @abc.abstractmethod
    def get_actions(self, observations):
        """Get actions given observations.

        Args:
            observations (np.ndarray): Observations from the environment.

        Returns:
            Tuple[np.ndarray, dict[str,np.ndarray]]: Actions and extra agent
                infos.

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
