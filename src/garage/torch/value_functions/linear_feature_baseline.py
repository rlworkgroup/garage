"""A value function based on a GaussianMLP model."""
import numpy as np
import torch
from torch import nn

from garage.torch.value_functions.value_function import ValueFunction
from garage.np.baselines import LinearFeatureBaseline


class LinearFeatureValueFunction(ValueFunction):
    """Linear Feature Value Function with Model.

    Args:
        env_spec (EnvSpec): Environment specification.
        name (str): The name of the value function.

    """

    def __init__(self,
                 env_spec,
                 name='LinearFeatureValueFunction'):
        super(LinearFeatureValueFunction, self).__init__(env_spec, name)

        self._baseline = LinearFeatureBaseline(env_spec)

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
        preds = self.forward(obs)
        return (returns-preds).sum()

    def fit(self, paths):
        return self._baseline.fit(paths=paths)

    # pylint: disable=arguments-differ
    def forward(self, paths):
        r"""Predict value based on paths.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(P, O*)`.

        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.

        """
        preds = torch.Tensor([self._baseline.predict(path) for path in paths])
        return preds
