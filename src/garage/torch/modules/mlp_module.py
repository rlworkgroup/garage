"""
MLP Module.

A Pytorch module composed only of a multi-layer perceptron (MLP), which maps
real-valued inputs to real-valued outputs.
"""

from torch import nn as nn
from torch.nn import functional as F  # NOQA


class MLPModule(nn.Module):
    """
    MLP Model.

    Args:
        input_dim (int) : Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    Return:
        The output torch.Tensor of the MLP
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes,
                 hidden_nonlinearity=F.relu,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._layer_normalization = layer_normalization
        self._layers = nn.ModuleList()

        prev_size = input_dim

        for size in hidden_sizes:
            layer = nn.Linear(prev_size, size)
            hidden_w_init(layer.weight)
            hidden_b_init(layer.bias)
            self._layers.append(layer)
            prev_size = size

        layer = nn.Linear(prev_size, output_dim)
        output_w_init(layer.weight)
        output_b_init(layer.bias)
        self._layers.append(layer)

    def forward(self, input_val):
        """Forward method."""
        x = input_val
        for layer in self._layers[:-1]:
            x = layer(x)
            if self._hidden_nonlinearity is not None:
                x = self._hidden_nonlinearity(x)
            if self._layer_normalization:
                x = nn.LayerNorm(x.shape[1])(x)

        x = self._layers[-1](x)
        if self._output_nonlinearity is not None:
            x = self._output_nonlinearity(x)

        return x
