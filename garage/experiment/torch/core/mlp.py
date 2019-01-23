import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes,
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_normal_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_normal_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        """
        MLP model.

        Args:
            input_dim: Dimension of input.
            output_dim: Dimension of the network output.
            hidden_sizes: Output dimension of dense layer(s).
            name: variable scope of the mlp.
            hidden_nonlinearity: Activation function for
                        intermediate dense layer(s).
            hidden_w_init: Initializer function for the weight
                        of intermediate dense layer(s).
            hidden_b_init: Initializer function for the bias
                        of intermediate dense layer(s).
            output_nonlinearity: Activation function for
                        output dense layer.
            output_w_init: Initializer function for the weight
                        of output dense layer(s).
            output_b_init: Initializer function for the bias
                        of output dense layer(s).
            layer_normalization: Bool for using layer normalization or not.

        Return:
            The output torch.Tensor of the MLP
        """
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.layer_normalization = layer_normalization

        last_size = input_dim
        for size in hidden_sizes:
            layer = nn.Linear(last_size, size)
            hidden_w_init(layer.weight)
            hidden_b_init(layer.bias)
            self.layers.append(layer)
            last_size = size

        layer = nn.Linear(last_size, output_dim)
        output_w_init(layer.weight)
        output_b_init(layer.bias)
        self.layers.append(layer)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.hidden_nonlinearity(layer(x))
            if self.layer_normalization:
                x = nn.LayerNorm(x.shape[1])(x)

        x = self.layers[-1](x)
        if self.output_nonlinearity:
            x = self.output_nonlinearity(x)

        return x
