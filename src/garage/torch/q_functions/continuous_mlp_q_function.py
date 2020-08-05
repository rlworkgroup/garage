"""This modules creates a continuous Q-function network."""

import torch

from garage.torch.modules import MLPModule


class ContinuousMLPQFunction(MLPModule):
    """Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, env_spec, **kwargs):
        """Initialize class with multiple attributes.

        Args:
            env_spec (EnvSpec): Environment specification.
            **kwargs: Keyword arguments.

        """
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        MLPModule.__init__(self,
                           input_dim=self._obs_dim + self._action_dim,
                           output_dim=1,
                           **kwargs)

    # pylint: disable=arguments-differ
    def forward(self, observations, actions):
        """Return Q-value(s).

        Args:
            observations (np.ndarray): observations.
            actions (np.ndarray): actions.

        Returns:
            torch.Tensor: Output value
        """
        return super().forward(torch.cat([observations, actions], 1))
