# pylint: disable=attribute-defined-outside-init
"""A recurrent network with LSTM for encoding context of RL tasks."""

import torch
from torch import nn as nn

from garage.torch.modules import MLPModule


class RecurrentEncoder(MLPModule):
    """This recurrent network encodes context of RL tasks.

    Context is stored in the terms of observation, action, and reward, and this
    network uses an MLP module followed by an LSTM model for encoding.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments. Refer MLPModule arguments.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = self._hidden_sizes[-1]
        # hidden dimension should be (task, 1, feat)
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))
        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            num_layers=1,
                            batch_first=True)

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method with LSTM.

        Args:
            input_val (torch.Tensor): Input values with shape
                (task, seq, feat).

        Returns:
            torch.Tensor: Output values.

        """
        task, seq, feat = input_val.size()
        out = input_val.view(task * seq, feat)

        # embed with MLP
        for _, layer in enumerate(self._layers):
            out = layer(out)
            if self._hidden_nonlinearity is not None:
                out = self._hidden_nonlinearity(out)
        out = out.view(task, seq, -1)

        # add LSTM before output layer
        # step through the entire sequence of LSTM all at once
        # out = all hidden states in the sequence
        # hn = last hidden state with gradients
        out, (hn,
              _) = self.lstm(out,
                             (self.hidden, torch.zeros(self.hidden.size())))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        output = self._output_layers[-1](out)
        if self._output_nonlinearity is not None:
            output = self._output_nonlinearity(output)

        return output

    def reset(self, num_tasks=1):
        """Reset task size in hidden dimensions.

        Args:
            num_tasks (int): Size of tasks.

        """
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)
