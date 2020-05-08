"""Base Policy."""
import abc

import torch


class Policy(abc.ABC, torch.nn.Module):
    """Policy base class.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Name of policy.

    """

    def __init__(self, env_spec, name):
        # pylint: disable=super-init-not-called
        # See issue #1141
        self._env_spec = env_spec
        self._name = name

    @abc.abstractmethod
    def get_action(self, observation):
        """Get a single action given an observation.

        Args:
            observation (torch.Tensor): Observation from the environment.

        Returns:
            tuple:
                * torch.Tensor: Predicted action.
                * dict:
                    * list[float]: Mean of the distribution
                    * list[float]: Log of standard deviation of the
                        distribution

        """

    @abc.abstractmethod
    def get_actions(self, observations):
        """Get actions given observations.

        Args:
            observations (torch.Tensor): Observations from the environment.

        Returns:
            tuple:
                * torch.Tensor: Predicted actions.
                * dict:
                    * list[float]: Mean of the distribution
                    * list[float]: Log of standard deviation of the
                        distribution

        """

    @property
    def observation_space(self):
        """The observation space for the environment.

        Returns:
            akro.Space: Observation space.

        """
        return self._env_spec.observation_space

    @property
    def action_space(self):
        """The action space for the environment.

        Returns:
            akro.Space: Action space.

        """
        return self._env_spec.action_space

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            bool: flag for vectorized

        """
        return False

    @property
    def name(self):
        """Name of policy.

        Returns:
            str: Name of policy

        """
        return self._name

    def get_param_values(self):
        """Get the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Returns:
            dict: The parameters (in the form of the state dictionary).

        """
        return self.state_dict()

    def set_param_values(self, state_dict):
        """Set the parameters to the policy.

        This method is included to ensure consistency with TF policies.

        Args:
            state_dict (dict): State dictionary.

        """
        self.load_state_dict(state_dict)
