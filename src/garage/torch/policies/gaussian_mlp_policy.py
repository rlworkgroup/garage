"""GaussianMLPPolicy."""
import torch

from garage.torch.modules import GaussianMLPModule
from garage.torch.policies import Policy


class GaussianMLPPolicy(Policy, GaussianMLPModule):
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
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        Policy.__init__(self, env_spec)
        GaussianMLPModule.__init__(self,
                                   input_dim=self._obs_dim,
                                   output_dim=self._action_dim,
                                   **kwargs)

    def forward(self, inputs):
        """Forward method."""
        return super().forward(torch.Tensor(inputs))

    def get_action(self, observation):
        """Get a single action given an observation."""
        with torch.no_grad():
            observation = observation.unsqueeze(0)
            dist = self.forward(observation)
            return (dist.rsample().squeeze(0).numpy(),
                    dict(mean=dist.mean.squeeze(0).numpy(),
                         log_std=(dist.variance**.5).log().squeeze(0).numpy()))

    def get_actions(self, observations):
        """Get actions given observations."""
        with torch.no_grad():
            dist = self.forward(observations)
            return (dist.rsample().numpy(),
                    dict(mean=dist.mean.numpy(),
                         log_std=(dist.variance**.5).log().numpy()))

    def log_likelihood(self, observation, action):
        """Get log likelihood given observations and action."""
        dist = self.forward(observation)
        return dist.log_prob(action)

    def get_entropy(self, observation):
        """Get entropy given observations."""
        dist = self.forward(observation)
        return dist.entropy()

    def reset(self, dones=None):
        """Reset the environment."""
        pass

    @property
    def vectorized(self):
        """Vectorized or not."""
        return True
