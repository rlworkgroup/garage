"""Exploration Policy API used by off-policy algorithms."""
import abc


# This should be an ABC inheritting from garage.Policy, but that doesn't exist
# yet.
class ExplorationPolicy(abc.ABC):
    """Policy that wraps another policy to add action noise.

    Args:
        policy (garage.Policy): Policy to wrap.

    """

    def __init__(self, policy):
        self.policy = policy

    @abc.abstractmethod
    def get_action(self, observation):
        """Return an action with noise.

        Args:
            observation (np.ndarray): Observation from the environment.

        Returns:
            np.ndarray: An action with noise.
            dict: Arbitrary policy state information (agent_info).

        """

    @abc.abstractmethod
    def get_actions(self, observations):
        """Return actions with noise.

        Args:
            observations (np.ndarray): Observation from the environment.

        Returns:
            np.ndarray: Actions with noise.
            List[dict]: Arbitrary policy state information (agent_info).

        """

    def reset(self, dones=None):
        """Reset the state of the exploration.

        Args:
            dones (List[bool] or numpy.ndarray or None): Which vectorization
                states to reset.

        """
        self.policy.reset(dones)

    def get_param_values(self):
        """Get parameter values.

        Returns:
            list or dict: Values of each parameter.

        """
        return self.policy.get_param_values()

    def set_param_values(self, params):
        """Set param values.

        Args:
            params (np.ndarray): A numpy array of parameter values.

        """
        self.policy.set_param_values(params)
