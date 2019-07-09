"""This modules creates a continuous Q-function network."""

import torch


class ContinuousNNQFunction:
    """
    Implements a module-agnostic Q-value network.

    It predicts the q value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, env_spec, nn_module, input_include_goal=False):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            nn_module (nn.Module): Neural network module in PyTorch.
        """
        self._env_spec = env_spec
        self._nn_module = nn_module
        if input_include_goal:
            self._obs_dim = env_spec.observation_space.flat_dim_with_keys(
                ['observation', 'desired_goal'])
        else:
            self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._action_bound = env_spec.action_space.high

    def get_qval(self, observations, actions):
        """Return Q value."""
        return self._nn_module.forward(torch.cat([observations, actions], 0))
