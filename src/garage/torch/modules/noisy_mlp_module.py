"""Noisy MLP Module."""

import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from garage.torch import NonLinearity


# pytorch v1.6 issue, see https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class NoisyMLPModule(nn.Module):
    """MLP with factorised gaussian parameter noise.

    See https://arxiv.org/pdf/1706.10295.pdf.

    This module creates a multilayered perceptron (MLP) in where the
    linear layers are replaced with :class:`~NoisyLinear` layers.
    See the docstring of :class:`~NoisyLinear` and the linked paper
    for more details.

    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable or torch.nn.Module): Activation function
            for intermediate dense layer(s). It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        sigma_naught (float): Hyperparameter that specifies the intial noise
            scaling factor. See the paper for details.
        std_noise (float): Standard deviation of the gaussian noise
            distribution to sample from.
        output_nonlinearity (callable or torch.nn.Module): Activation function
            for output dense layer. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes,
                 hidden_nonlinearity=F.relu,
                 sigma_naught=0.5,
                 std_noise=1.,
                 output_nonlinearity=None):
        super().__init__()
        self._layers = nn.ModuleList()
        self._noisy_layers = []

        prev_size = input_dim
        for size in hidden_sizes:
            hidden_layers = nn.Sequential()
            linear_layer = NoisyLinear(prev_size, size, sigma_naught,
                                       std_noise)
            self._noisy_layers.append(linear_layer)
            hidden_layers.add_module('linear', linear_layer)

            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity',
                                         NonLinearity(hidden_nonlinearity))

            self._layers.append(hidden_layers)
            prev_size = size

        self._output_layers = nn.ModuleList()
        output_layer = nn.Sequential()
        linear_layer = NoisyLinear(prev_size, output_dim, sigma_naught,
                                   std_noise)
        self._noisy_layers.append(linear_layer)
        output_layer.add_module('linear', linear_layer)

        if output_nonlinearity:
            output_layer.add_module('non_linearity',
                                    NonLinearity(output_nonlinearity))

        self._output_layers.append(output_layer)

    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = input_val
        for layer in self._layers:
            x = layer(x)

        return self._output_layers[0](x)

    def set_deterministic(self, deterministic):
        """Set whether or not noise is applied.

        This is useful when determinstic evaluation of
        a policy is desired. Determinism is off by default.

        Args:
            deterministic (bool): If False, noise is applied, else
                it is not.
        """
        for layer in self._noisy_layers:
            layer.set_deterministic(deterministic)


class NoisyLinear(nn.Module):
    r"""Noisy linear layer with Factorised Gaussian noise.

    See https://arxiv.org/pdf/1706.10295.pdf.

    Each NoisyLinear layer applies the following transformation

    :math:`y = (\mu^w + \sigma^w  \odot \epsilon ^w) + \mu^b + \sigma^b \odot
    \epsilon^b`

    where :math:`\mu^w, \mu^b, \sigma^w, and \sigma^b` are learned parameters
    and :math:`\epislon^w, \epsilon^b` are zero-mean gaussian noise samples.

    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        sigma_naught (float): Hyperparameter that specifies the intial noise
            scaling factor. See the paper for details.
        std_noise (float): Standard deviation of the gaussian noise
            distribution to sample from.
    """

    def __init__(self, input_dim, output_dim, sigma_naught=0.5, std_noise=1.):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._sigma_naught = sigma_naught
        self._std_noise = std_noise
        self._deterministic = False

        self._output_dim = output_dim

        self._weight_mu = nn.Parameter(torch.FloatTensor(
            output_dim, input_dim))
        self._weight_sigma = nn.Parameter(
            torch.FloatTensor(output_dim, input_dim))

        self._bias_mu = nn.Parameter(torch.FloatTensor(output_dim))
        self._bias_sigma = nn.Parameter(torch.FloatTensor(output_dim))

        # epsilon noise
        self.register_buffer('weight_epsilon',
                             torch.FloatTensor(output_dim, input_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_dim))

        self.reset_parameters()

    def set_deterministic(self, deterministic):
        """Set whether or not noise is applied.

        This is useful when determinstic evaluation of
        a policy is desired. Determinism is off by default.

        Args:
            deterministic (bool): If False, noise is applied, else
                it is not.
        """
        self._deterministic = deterministic

    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            torch.Tensor: Output value
        """
        if self._deterministic:
            return F.linear(input_value, self._weight_mu, self._bias_mu)

        self._sample_noise()
        w = self._weight_mu + self._weight_sigma.mul(
            Variable(self.weight_epsilon))
        b = self._bias_mu + self._bias_sigma.mul(Variable(self.bias_epsilon))
        return F.linear(input_value, w, b)

    def reset_parameters(self):
        """Reset all learnable parameters."""
        mu_range = 1 / math.sqrt(self._weight_mu.size(1))

        self._weight_mu.data.uniform_(-mu_range, mu_range)
        self._weight_sigma.data.fill_(self._sigma_naught /
                                      math.sqrt(self._weight_sigma.size(1)))

        self._bias_mu.data.uniform_(-mu_range, mu_range)
        self._bias_sigma.data.fill_(self._sigma_naught /
                                    math.sqrt(self._bias_sigma.size(0)))

    def _sample_noise(self):
        r"""Sample and assign new values for :math:`\epsilon`."""
        in_noise = self._get_noise(self._input_dim)
        out_noise = self._get_noise(self._output_dim)
        self.weight_epsilon.copy_(out_noise.ger(in_noise))
        self.bias_epsilon.copy_(self._get_noise(self._output_dim))

    def _get_noise(self, size):
        """Retrieve scaled zero-mean gaussian noise.

        Args:
            size (int): size of the noise vector.

        Returns:
            torch.Tensor: noise vector of the specified size.
        """
        x = torch.normal(torch.zeros(size), self._std_noise * torch.ones(size))
        return x.sign().mul(x.abs().sqrt())
