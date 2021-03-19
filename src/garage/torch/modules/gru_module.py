"""GRU Module."""
import copy

import torch
from torch import nn
from torch.autograd import Variable

from garage.experiment import deterministic
from garage.torch import global_device, NonLinearity


# pytorch v1.6 issue, see https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
# pylint: disable=unused-argument
class GRUModule(nn.Module):
    """Gated Recurrent Unit (GRU) model in pytorch.

    Args:
        input_dim (int): Dimension of the network input.
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
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        hidden_nonlinearity=nn.Tanh,
        hidden_w_init=nn.init.xavier_uniform_,
        hidden_b_init=nn.init.zeros_,
        layer_normalization=False,
    ):
        super().__init__()
        self._layers = nn.Sequential()
        self.hidden_dim = hidden_dim
        self._gru_cell = nn.GRUCell(input_dim, hidden_dim)
        hidden_w_init(self._gru_cell.weight_ih)
        hidden_w_init(self._gru_cell.weight_hh)
        hidden_b_init(self._gru_cell.bias_ih)
        hidden_b_init(self._gru_cell.bias_hh)
        self.hidden_nonlinearity = NonLinearity(hidden_nonlinearity)

        self._layers.add_module("activation", self.hidden_nonlinearity)
        if layer_normalization:
            self._layers.add_module("layer_normalization", nn.LayerNorm(hidden_dim))

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim) shape.

        Returns:
            torch.Tensor: Output values with (N, *, hidden_dim) shape.

        """
        if len(input_val.size()) == 2:
            input_val = input_val.unsqueeze(0)
        h0 = Variable(
            torch.zeros(input_val.size(0), self.hidden_dim)).to(global_device())
        outs = []
        hn = h0
        for seq in range(input_val.size(1)):
            hn = self._gru_cell(input_val[:, seq, :], hn)
            outs.append(hn)
        out = outs[-1].squeeze(dim=1)
        out = self._layers(out)
        outs = torch.stack(outs)
        return out