""" A recurrent network with LSTM for encoding context of RL tasks."""

import torch
from torch import nn as nn

from garage.torch.modules import MLPModule


class RecurrentEncoder(MLPModule):
    """
    This recurrent network encodes context (observation, action, reward) of RL
    tasks using an MLP module followed by an LSTM model. 

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
    """

    def __init__(self,
                 *args,
                 **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hidden_dim = self._hidden_sizes[-1]
        # hidden dimension should be (task, 1, feat)
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))
        self.lstm = nn.LSTM(self.hidden_dim, 
                            self.hidden_dim, 
                            num_layers=1, 
                            batch_first=True)

    def forward(self, input_val):
        # input dimension should be (task, seq, feat)
        task, seq, feat = input_val.size()
        out = input_val.view(task * seq, feat)

        # embed with MLP
        for i, layer in enumerate(self._layers):
            out = layer(out)
            if self._hidden_nonlinearity is not None:
                out = self._hidden_nonlinearity(out)
        out = out.view(task, seq, -1)

        # add lstm before output layer
        # step through the entire sequence of lstm all at once
        # out = all hidden states in the sequence
        # hn = last hidden state with gradients
        out, (hn, cn) = self.lstm(out, (self.hidden, 
            torch.zeros(self.hidden.size())))
        self.hidden = hn
        # take the last hidden state to predict z
        out = out[:, -1, :]

        # output layer
        output = self._output_layers[-1](out)
        if self._output_nonlinearity is not None:
            output = self._output_nonlinearity(output)

        return output

    def reset(self, num_tasks=1):
        """reset hidden dimensions"""
        # reset task dimension in hidden from 1 to num_task
        self.hidden = self.hidden.new_full(
            (1, num_tasks, self.hidden_dim), 0)