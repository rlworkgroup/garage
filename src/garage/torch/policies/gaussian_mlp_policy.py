"""GaussianMLPPolicy."""
import numpy as np
import torch

from garage.torch.modules import GaussianMLPModule
from garage.torch.policies import Policy


class GaussianMLPPolicy(GaussianMLPModule, Policy):
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

    def __init__(self, env_spec, **kwargs):
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        GaussianMLPModule.__init__(self,
                                   input_dim=self._obs_dim,
                                   output_dim=self._action_dim,
                                   **kwargs)

    def forward(self, inputs):
        """Forward method."""
        return super().forward(inputs)

    def get_action(self, observation):
        """Get a single action given an observation."""
        with torch.no_grad():
            observation = observation.unsqueeze(0)
            dist = self.forward(observation)
            std = dist.variance**0.5
            return dist.rsample().squeeze(0).numpy(), dict(
                mean=dist.mean.squeeze(0), log_std=np.log(std).squeeze(0))

    def get_actions(self, observations):
        """Get actions given observations."""
        with torch.no_grad():
            dist = self.forward(observations)
            std = dist.variance**0.5
            return dist.rsample().detach().numpy(), dict(mean=dist.mean,
                                                         log_std=np.log(std))
