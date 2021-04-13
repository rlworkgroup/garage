"""A Discrete QFunction-derived policy.

This policy chooses the action that yields to the largest Q-value.
"""
import numpy as np
import torch

from garage.torch import np_to_torch
from garage.torch.policies.policy import Policy


class DiscreteQFArgmaxPolicy(Policy):
    """Policy that derives its actions from a learned Q function.

    The action returned is the one that yields the highest Q value for
    a given state, as determined by the supplied Q function.

    Args:
        qf (object): Q network.
        env_spec (EnvSpec): Environment specification.
        name (str): Name of this policy.
    """

    def __init__(self, qf, env_spec, name='DiscreteQFArgmaxPolicy'):
        super().__init__(env_spec, name)
        self._qf = qf

    # pylint: disable=arguments-differ
    def forward(self, observations):
        """Get actions corresponding to a batch of observations.

        Args:
            observations(torch.Tensor): Batch of observations of shape
                :math:`(N, O)`. Observations should be flattened even
                if they are images as the underlying Q network handles
                unflattening.

        Returns:
            torch.Tensor: Batch of actions of shape :math:`(N, A)`
        """
        qs = self._qf(observations)
        return torch.argmax(qs, dim=1)

    def get_action(self, observation):
        """Get a single action given an observation.

        Args:
            observation (np.ndarray): Observation with shape :math:`(O, )`.

        Returns:
            torch.Tensor: Predicted action with shape :math:`(A, )`.
            dict: Empty since this policy does not produce a distribution.
        """
        act, info = self.get_actions(np.expand_dims(observation, axis=0))
        return act[0], info

    def get_actions(self, observations):
        """Get actions given observations.

        Args:
            observations (np.ndarray): Batch of observations, should
                have shape :math:`(N, O)`.

        Returns:
            torch.Tensor: Predicted actions. Tensor has shape :math:`(N, A)`.
            dict: Empty since this policy does not produce a distribution.
        """
        with torch.no_grad():
            return self(np_to_torch(observations)).cpu().numpy(), dict()
