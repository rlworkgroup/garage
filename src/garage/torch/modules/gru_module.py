"""GRU in Pytorch."""
import torch
from torch import nn
from torch.autograd import Variable


class GRUModule(nn.Module):

    def __init__(
            self,
            input_dim,
            hidden_dim,
            # hidden_nonlinearity,
            layer_dim,
            output_dim,
            bias=True):
        super().__init__()
        self._hidden_dim = hidden_dim
        # Number of hidden layers
        self._layer_dim = layer_dim
        self._gru_cell = nn.GRUCell(input_dim, hidden_dim)
        # self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        self._fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, *input):
        # input = input[0]
        input = Variable(input[0].view(-1, input[0].size(0), input[0].size(1)))

        # Initialize hidden state with zeros
        if torch.cuda.is_available():
            h0 = Variable(
                torch.zeros(self._layer_dim, input.size(0),
                            self._hidden_dim).cuda())
        else:
            h0 = Variable(
                torch.zeros(self._layer_dim, input.size(0), self._hidden_dim))

        outs = []
        hn = h0[0, :, :]

        for seq in range(input.size(1)):
            hn = self._gru_cell(input[:, seq, :], hn)
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self._fc(out)
        outs = torch.stack(outs)  # convert list of tensors to tensor
        outs = self._fc(outs)
        # out.size() --> 100, 10
        # return out
        return outs, out, hn, h0
