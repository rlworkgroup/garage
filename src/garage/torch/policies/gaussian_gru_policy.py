""""Gaussian GRU Policy."""
import torch
from torch import nn
from torch.distributions.independent import Independent

from garage.torch.modules import GaussianGRUModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class GaussianGRUPolicy(StochasticPolicy):
    """Gaussian GRU Policy.
    """

    def __init__(
            self,
            env_spec,
            hidden_dim=(32, 32),
            hidden_nonlinearity=torch.tanh,
            hidden_w_init=nn.init.xavier_uniform_,
            hidden_b_init=nn.init.zeros_,
            output_nonlinearity=None,
            output_w_init=nn.init.xavier_uniform_,
            output_b_init=nn.init.zeros_,
            learn_std=True,
            init_std=1.0,
            #  min_std=1e-6,
            #  max_std=None,
            std_parameterization='exp',
            layer_normalization=False,
            name='GaussianGRUPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._module = GaussianGRUModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            hidden_dim=hidden_dim,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_std=learn_std,
            init_std=init_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization)

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """

        dist = self._module(observations)
        dist = dist[0]
        assert isinstance(dist, torch.distributions.independent.Independent)
        assert dist.event_shape == torch.Size((1, ))
        assert dist.batch_shape == torch.Size((1, ))
        return (dist, dict(mean=dist.mean, log_std=(dist.variance**.5).log()))
