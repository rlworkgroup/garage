"""
This modules creates a deterministic policy network.

A neural network can be used as policy method in different RL algorithms.
It accepts an observation of the environment and predicts an action.
"""
import torch

from garage.torch.modules import MLPModule
from garage.torch.policies import Policy


class DeterministicMLPPolicy(MLPModule, Policy):
    """
    Implements a deterministic policy network.

    The policy network selects action based on the state of the environment.
    It uses a PyTorch neural network module to fit the function of pi(s).
    """

    def __init__(self, env_spec, **kwargs):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            nn_module (nn.Module): Neural network module in PyTorch.
        """
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        MLPModule.__init__(self,
                           input_dim=self._obs_dim,
                           output_dim=self._action_dim,
                           **kwargs)

    def forward(self, input_val):
        """Forward method."""
        input_val = torch.FloatTensor(input_val)
        return super().forward(input_val)

    def get_action(self, observation):
        """Return a single action."""
        with torch.no_grad():
            x = self.forward(observation.unsqueeze(0))
            return x.squeeze(0).numpy(), dict()

    def get_actions(self, observations):
        """Return multiple actions."""
        with torch.no_grad():
            x = self.forward(observations)
            return x.numpy(), dict()

    def reset(self, dones=None):
        """Reset policy."""
        pass
