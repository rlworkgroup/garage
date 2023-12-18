"""Gaussian GRU Module."""
import abc

import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.independent import Independent
import torch.nn.functional as F

from garage.torch import global_device, set_gpu_mode
from garage.torch.distributions import TanhNormal
from garage.torch.modules.gru_module import GRUModule

class GaussianGRUBaseModule(nn.Module):
    """Gaussian GRU Module.

    A model represented by a Gaussian distribution
    which is parameterized by a Gated Recurrent Unit (GRU).
    """

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=(32, 32),
            hidden_nonlinearity=torch.tanh,
            hidden_w_init=nn.init.xavier_uniform_,
            hidden_b_init=nn.init.zeros_,
            recurrent_nonlinearity=torch.sigmoid,
            recurrent_w_init=nn.init.xavier_uniform_,
            output_nonlinearity=None,
            output_w_init=nn.init.xavier_uniform_,
            output_b_init=nn.init.zeros_,
            learn_std=True,
            init_std=1.0,
            std_parameterization='exp',
            layer_normalization=False,
            normal_distribution_cls=Normal):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._recurrent_nonlinearity = recurrent_nonlinearity
        self._recurrent_w_init = recurrent_w_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._learn_std = learn_std
        self._std_parameterization = std_parameterization,
        self._layer_normalization = layer_normalization
        self._norm_dist_class = normal_distribution_cls

        self.continuous_action_space = True# continuous_action_space 
        self.log_std_dev = nn.Parameter(init_std * torch.ones(( self._output_dim), dtype=torch.float), requires_grad=self._learn_std)
        self.covariance_eye = torch.eye(int(self._output_dim)).unsqueeze(0)

        init_std_param = torch.Tensor([init_std]).log()
        if self._learn_std:
            self._init_std = torch.nn.Parameter(init_std_param)
        else:
            self._init_std = init_std_param
            self.register_buffer('init_std', self._init_std)

    def to(self, *args, **kwargs):
        """Move the module to the specified device.

        Args:
            *args: args to pytorch to function.
            **kwargs: keyword args to pytorch to function.

        """
        super().to(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._init_std, torch.nn.Parameter):
            self._init_std = buffers['init_std']

    @abc.abstractmethod
    def _get_mean_and_log_std(self, *inputs):
        pass

    def forward(self, *inputs, terminal=None):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.distributions.independent.Independent: Independent
                distribution.

        """
        if torch.cuda.is_available():
            set_gpu_mode(True)
        else:
            set_gpu_mode(False)
        device = global_device()

        _ , _ , _ , _, policy_logits_out, _ = self._get_mean_and_log_std(*inputs)
        if self.continuous_action_space:
            cov_matrix = self.covariance_eye.to(device).expand(self._input_dim, self._output_dim, self._output_dim) * torch.exp(self._init_std.to(device))
            # We define the distribution on the CPU since otherwise operations fail with CUDA illegal memory access error.
            policy_dist = torch.distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"), cov_matrix.to("cpu"))
        else:
            policy_dist = torch.distributions.Categorical(F.softmax(policy_logits_out, dim=1).to("cpu"))
        
        # if not isinstance(policy_dist, TanhNormal):
        # #     # Makes it so that a sample from the distribution is treated as a
        # #     # single sample and not dist.batch_shape samples.
        #     policy_dist = Independent(policy_dist, 1)
        return policy_dist

        (mean, step_mean, log_std, step_log_std, step_hidden,
         hidden_init) = self._get_mean_and_log_std(*inputs)

        if self._std_parameterization == 'exp':
            std = log_std.exp()
        else:
            std = log_std.exp().exp().add(1.).log()
        dist = self._norm_dist_class(mean, std)
        if not isinstance(dist, TanhNormal):
            # Makes it so that a sample from the distribution is treated as a
            # single sample and not dist.batch_shape samples.
            dist = Independent(dist, 1)

        return dist
        # return (dist, step_mean, step_log_std, step_hidden, hidden_init)


class GaussianGRUModule(GaussianGRUBaseModule):
    """GaussianMLPModule that mean and std share the same network.

    """

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=(32, 32),
            hidden_nonlinearity=torch.tanh,
            hidden_w_init=nn.init.xavier_uniform_,
            hidden_b_init=nn.init.zeros_,
            recurrent_nonlinearity=torch.sigmoid,
            recurrent_w_init=nn.init.xavier_uniform_,
            output_nonlinearity=None,
            output_w_init=nn.init.xavier_uniform_,
            output_b_init=nn.init.zeros_,
            #  hidden_state_init=nn.init.zeros_,
            learn_std=True,
            init_std=1.0,
            std_parameterization='exp',
            layer_normalization=False,
            normal_distribution_cls=Normal):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
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
            layer_normalization=layer_normalization,
            normal_distribution_cls=normal_distribution_cls)

        self._mean_gru_module = GRUModule(input_dim=self._input_dim,
                                          output_dim=self._output_dim,
                                          hidden_dim=self._hidden_dim,
                                          layer_dim=1)

    def _get_mean_and_log_std(self, *inputs):
        """Get mean and std of Gaussian distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The mean of Gaussian distribution.
            torch.Tensor: The variance of Gaussian distribution.

        """
        assert len(inputs) == 1
        (mean_outputs, step_mean_outputs, step_hidden,
         hidden_init_var) = self._mean_gru_module(*inputs)

        uncentered_log_std = torch.zeros(inputs[0].size(-1)) + self._init_std

        uncentered_step_log_std = torch.zeros(
            inputs[0].size(-1)) + self._init_std

        return (mean_outputs, step_mean_outputs, uncentered_log_std,
                uncentered_step_log_std, step_hidden, hidden_init_var)
