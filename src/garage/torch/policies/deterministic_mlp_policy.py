"""This modules creates a deterministic policy network.

A neural network can be used as policy method in different RL algorithms.
It accepts an observation of the environment and predicts an action.
"""
import torch

from garage.torch.modules import MLPModule
from garage.torch.policies import Policy


class DeterministicMLPPolicy(MLPModule, Policy):
    """Implements a deterministic policy network.

    The policy network selects action based on the state of the environment.
    It uses a PyTorch neural network module to fit the function of pi(s).
    """

    def __init__(self, env_spec, name='DeterministicMLPPolicy', **kwargs):
        """Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            name (str): Policy name.
            kwargs : Additional keyword arguments passed to the MLPModule.

        """
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        Policy.__init__(self, env_spec, name)

        MLPModule.__init__(self,
                           input_dim=self._obs_dim,
                           output_dim=self._action_dim,
                           **kwargs)

    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): values to compute

        Returns:
            torch.Tensor: computed values

        """
        input_val = torch.FloatTensor(input_val)
        return super().forward(input_val)

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
        with torch.no_grad():
            x = self.forward(observation.unsqueeze(0))
            return x.squeeze(0).numpy(), dict()

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
        with torch.no_grad():
            x = self.forward(observations)
            return x.numpy(), dict()

    def reset(self, dones=None):
        """Reset the environment.

        Args:
            dones (numpy.ndarray): Reset values

        """
