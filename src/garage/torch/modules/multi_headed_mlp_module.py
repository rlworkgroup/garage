"""MultiHeadedMLPModule."""
import torch.nn as nn


class MultiHeadedMLPModule(nn.Module):
    """
    MultiHeadedMLPModule Model.

    Args:
        n_heads (int) : Number of different output layers
        input_dim (int) : Dimension of the network input.
        output_dims (int, list, or tuple): Dimension of the network output.
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
        output_nonlinearity (callable, list, or tuple): Activation function for
            output dense layer. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_head
        output_w_init (callable, list, or tuple): Initializer function for the
            weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        output_b_init (callable, list, or tuple): Initializer function for the
            bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        layer_normalization (bool): Bool for using layer normalization or not.

    Return:
        The list of output torch. Tensor of the MLP
    """

    def __init__(self,
                 n_heads,
                 input_dim,
                 output_dims,
                 hidden_sizes,
                 hidden_nonlinearity=nn.ReLU,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearities=None,
                 output_w_inits=nn.init.xavier_normal_,
                 output_b_inits=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__()

        self._layers = nn.ModuleList()

        if n_heads < 1:
            raise ValueError(
                'n_head should be greater than or equal to 1 but {} provided.'.
                format(n_heads))

        output_dims = self._check_hidden_layer_parameter(
            'output_dims', output_dims, n_heads)
        output_nonlinearities = self._check_hidden_layer_parameter(
            'output_nonlinearities', output_nonlinearities, n_heads)
        output_w_inits = self._check_hidden_layer_parameter(
            'output_w_inits', output_w_inits, n_heads)
        output_b_inits = self._check_hidden_layer_parameter(
            'output_b_inits', output_b_inits, n_heads)

        prev_size = input_dim
        for size in hidden_sizes:
            layer = nn.Sequential()
            if layer_normalization:
                layer.add_module('layer_normalization',
                                 nn.LayerNorm(prev_size))

            linear_layer = nn.Linear(prev_size, size)
            hidden_w_init(linear_layer.weight)
            hidden_b_init(linear_layer.bias)
            layer.add_module('linear', linear_layer)

            if hidden_nonlinearity:
                layer.add_module('non_linearity', hidden_nonlinearity)

            self._layers.append(layer)
            prev_size = size

        self._output_layers = nn.ModuleList()
        for i in range(n_heads):
            layer = nn.Sequential()

            linear_layer = nn.Linear(prev_size, output_dims[i])
            output_w_inits[i](linear_layer.weight)
            output_b_inits[i](linear_layer.bias)
            layer.add_module('linear', linear_layer)

            if output_nonlinearities[i]:
                layer.add_module('non_linearity', output_nonlinearities[i])

            self._output_layers.append(layer)

    def _check_hidden_layer_parameter(self, var_name, var, n_heads):
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                var = var * n_heads
            elif len(var) != n_heads:
                msg = '{} should be either an integer or' \
                      ' a collection of length n_heads ({}), but {} provided.'
                raise ValueError(msg.format(var_name, n_heads, var))
        else:
            var = [var] * n_heads

        return var

    def forward(self, input_val):
        """Forward method."""
        x = input_val
        for layer in self._layers:
            x = layer(x)

        return [layer(x) for layer in self._output_layers]
