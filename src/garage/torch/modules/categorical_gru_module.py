"""Categorical GRU Module.

A model represented by a Categorical distribution
which is parameterized by a Gated Recurrent Unit (GRU)
followed a multilayer perceptron (MLP).
"""
import torch
from torch import nn
from torch.distributions import Categorical

from garage.torch.modules.gru_module import GRUModule
from garage.torch import global_device

class CategoricalGRUModule(nn.Module):
    """Categorical GRU Model.
    A model represented by a Categorical distribution
    which is parameterized by a gated recurrent unit (GRU)
    followed by a fully-connected layer.
    
    Args:
        input_dim (int): Dimension of the network input.
        output_dim (int): Dimension of the network output.
        hidden_dim (int): Hidden dimension for GRU cell.
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
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        hidden_nonlinearity=nn.Tanh,
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        output_nonlinearity=None,
        output_w_init=nn.init.xavier_uniform_,
        output_b_init=nn.init.zeros_,
        layer_normalization=False,
    ):
        super().__init__()

        self._gru_module = GRUModule(
            input_dim,
            hidden_dim,
            hidden_nonlinearity,
            hidden_w_init,
            hidden_b_init,
            layer_normalization,
        )

        self._linear_layer = nn.Sequential()
        hidden_layer = nn.Linear(hidden_dim, output_dim)
        output_w_init(hidden_layer.weight)
        output_b_init(hidden_layer.bias)
        self._linear_layer.add_module("output", hidden_layer)
        if output_nonlinearity:
            self._linear_layer.add_module(
                "output_activation", NonLinearity(output_nonlinearity)
            )

    def forward(self, *inputs):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.distributions.Categorical: Policy distribution.

        """
        assert len(inputs) == 1
        gru_output = self._gru_module(inputs[0])
        fc_output = self._linear_layer(gru_output)
        dist = Categorical(logits=fc_output.unsqueeze(0))
        return dist
