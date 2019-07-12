"""GaussianMLPPolicy."""
from torch import nn

from garage.torch.policies import Policy


class GaussianMLPPolicy(nn.Module, Policy):
    """
    GaussianMLPPolicy.

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        module : GaussianMLPModule to make prediction based on a gaussian
        distribution.
    :return:

    """

    def __init__(self, env_spec, module):
        nn.Module.__init__(self)
        Policy.__init__(self, env_spec)

        self._module = module

    def forward(self, inputs):
        """Forward method."""
        return self._module(inputs)

    def get_actions(self, observations):
        """Get actions given observations."""
        dist = self.forward(observations)
        return dist.rsample().detach().numpy()
