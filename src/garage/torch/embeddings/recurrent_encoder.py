# pylint: disable=attribute-defined-outside-init
"""A recurrent network with LSTM for encoding context of RL tasks."""

import torch
from torch import nn

from garage.torch.modules import MLPModule


class RecurrentEncoder(MLPModule):
    """This recurrent network encodes context of RL tasks.

    Context is stored in the terms of observation, action, and reward, and this
    network uses an MLP module followed by an LSTM model for encoding it.

    Args:
        *args: MLPModule arguments.
        **kwargs: MLPModule arguments including:
            input_dim (int) : Dimension of the network input.
            output_dim (int): Dimension of the network output.
            hidden_sizes (list[int]): Output dimension of dense layer(s).
                For example, (32, 32) means this MLP consists of two
                hidden layers, each with 32 hidden units.
            hidden_nonlinearity (callable or torch.nn.Module): Activation
                function for intermediate dense layer(s). It should return a
                torch.Tensor.Set it to None to maintain a linear activation.
            hidden_w_init (callable): Initializer function for the weight
                of intermediate dense layer(s). The function should return a
                torch.Tensor.
            hidden_b_init (callable): Initializer function for the bias
                of intermediate dense layer(s). The function should return a
                torch.Tensor.
            output_nonlinearity (callable or torch.nn.Module): Activation
                function for output dense layer. It should return a
                torch.Tensor. Set it to None to maintain a linear activation.
            output_w_init (callable): Initializer function for the weight
                of output dense layer(s). The function should return a
                torch.Tensor.
            output_b_init (callable): Initializer function for the bias
                of output dense layer(s). The function should return a
                torch.Tensor.
            layer_normalization (bool): Bool for using layer normalization or
                not.


    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hidden_dim = self._hidden_sizes[-1]
        # hidden dimension should be (task, 1, feat)
        self.register_buffer('hidden', torch.zeros(1, 1, self._hidden_dim))
        self._lstm = nn.LSTM(self._hidden_dim,
                             self._hidden_dim,
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
        for layer in self._layers:
            out = layer(out)
            if self._hidden_nonlinearity is not None:
                out = self._hidden_nonlinearity(out)
        out = out.view(task, seq, -1)

        # add LSTM before output layer
        # step through the entire sequence of LSTM all at once
        # out = all hidden states in the sequence
        # hn = last hidden state with gradients
        out, (hn,
              _) = self._lstm(out,
                              (self.hidden, torch.zeros(self.hidden.size())))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        output = self._output_layers[-1](out)
        if self._output_nonlinearity is not None:
            output = self._output_nonlinearity(output)

        output = output.view(task, -1, self._output_dim)

        return output

    def reset(self, num_tasks=1):
        """Reset task size in hidden dimensions.

        Args:
            num_tasks (int): Size of tasks.

        """
        self.hidden = self.hidden.new_full((1, num_tasks, self._hidden_dim), 0)

    def detach_hidden(self):
        """Disable backprop through hidden."""
        self.hidden = self.hidden.detach()
