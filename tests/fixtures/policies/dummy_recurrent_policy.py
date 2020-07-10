"""Dummy Recurrent Policy for algo tests."""
import numpy as np

from garage.np.policies import Policy


class DummyRecurrentPolicy(Policy):
    """Dummy Recurrent Policy.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    """

    def __init__(
        self,
        env_spec,
    ):
        super().__init__(env_spec=env_spec)
        self.params = []
        self.param_values = np.random.uniform(-1, 1, 1000)

    def get_action(self, observation):
        """Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Predicted action.
            dict: Distribution parameters. Empty because no distribution is
                used.

        """
        return self.action_space.sample(), dict()

    def get_params_internal(self):
        """Return a list of policy internal params.

        Returns:
            list: Policy parameters.

        """
        return self.params

    def get_param_values(self):
        """Return values of params.

        Returns:
            np.ndarray: Policy parameters values.

        """
        return self.param_values
