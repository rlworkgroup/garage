"""GRU in Pytorch."""
import torch
from torch import nn
from torch.autograd import Variable


class GRUModule(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,
                 bias=True):
        super().__init__()
        # Hidden dimensions
        self._hidden_dim = hidden_dim
        # Number of hidden layers
        self._layer_dim = layer_dim
        self._gru_cell = nn.GRUCell(input_dim, hidden_dim)
        # self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)
        self._fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        #print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = Variable(
                torch.zeros(self._layer_dim, x.size(0),
                            self._hidden_dim).cuda())
        else:
            h0 = Variable(
                torch.zeros(self._layer_dim, x.size(0), self._hidden_dim))

        outs = []
        hn = h0[0, :, :]

        for seq in range(x.size(1)):
            hn = self._gru_cell(x[:, seq, :], hn)
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self._fc(out)
        # out.size() --> 100, 10
        # return out
        return outs, out, hn, h0
