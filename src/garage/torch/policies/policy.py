"""Base Policy."""
import abc

import torch

from garage.np.policies import Policy as BasePolicy


class Policy(torch.nn.Module, BasePolicy, abc.ABC):
    """Policy base class.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Name of policy.

    """

    def __init__(self, env_spec, name):
        super().__init__()
        self._env_spec = env_spec
        self._name = name

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

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec

    @property
    def name(self):
        """Name of policy.

        Returns:
            str: Name of policy

        """
        return self._name
