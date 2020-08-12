"""Base class for all baselines."""
import abc

import torch.nn as nn


class ValueFunction(abc.ABC, nn.Module):
    """Base class for all baselines.

    Args:
        env_spec (EnvSpec): Environment specification.
        name (str): Value function name, also the variable scope.

    """

    def __init__(self, env_spec, name):
        super(ValueFunction, self).__init__()

        self._mdp_spec = env_spec
        self.name = name

    @abc.abstractmethod
    def compute_loss(self, obs, returns):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """
