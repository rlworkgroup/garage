"""Uniform control policy."""
from garage.tf.policies.policy import Policy


class UniformControlPolicy(Policy):
    """Policy that output random action uniformly.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    """

    def __init__(self, env_spec):
        self._env_spec = env_spec

    def get_action(self, observation):
        """Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Action
            dict: Predicted action and agent information. It returns an empty
                dict since there is no parameterization.

        """
        return self.action_space.sample(), dict()

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Actions
            dict: Predicted action and agent information. It returns an empty
                dict since there is no parameterization.

        """
        return self.action_space.sample_n(len(observations)), dict()
