"""Categorical CNN Module.

A model represented by a categorical distribution
which is parameterized by a convolutional neural network (CNN)
followed a multilayer perceptron (MLP).
"""
import torch
from torch import nn
from torch.distributions import Categorical

from garage.torch.modules.cnn_module import CNNBaseModule
from garage.torch.modules.mlp_module import MLPModule


class CategoricalCNNModule(nn.Module):
    """Categorical CNN Model.

    A model represented by a Categorical distribution
    which is parameterized by a convolutional neural network (CNN) followed
    by a multilayer perceptron (MLP).

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        kernel_sizes (tuple[int]): Dimension of the conv filters.
            For example, (3, 5) means there are two convolutional layers.
            The filter for first layer is of dimension (3 x 3)
            and the second one is of dimension (5 x 5).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_sizes (tuple[int]): Number of output channels for CNN.
            For example, (3, 32) means there are two convolutional layers.
            The filter for the first conv layer outputs 3 channels
            and the second one outputs 32 channels.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        paddings (tuple[int]):  Zero-padding added to both sides of the input
        padding_mode (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        max_pool (bool): Bool for using max-pooling or not.
        pool_shapes (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            shape (2, 2).
        pool_strides (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
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
            kernel_sizes,
            strides,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=torch.tanh,
            hidden_w_init=nn.init.xavier_uniform_,
            hidden_b_init=nn.init.zeros_,
            paddings=0,
            padding_mode='zeros',
            max_pool=False,
            pool_shapes=None,
            pool_strides=1,
            output_nonlinearity=None,
            output_w_init=nn.init.xavier_uniform_,
            output_b_init=nn.init.zeros_,
            layer_normalization=False,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._action_dim = output_dim
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._paddings = paddings
        self._padding_mode = padding_mode
        self._max_pool = max_pool
        self._pool_shapes = pool_shapes
        self._pool_strides = pool_strides
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        self._cnn_module = CNNBaseModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            kernel_sizes=self._kernel_sizes,
            strides=self._strides,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            paddings=self._paddings,
            padding_mode=self._padding_mode,
            max_pool=self._max_pool,
            pool_shapes=self._pool_shapes,
            pool_strides=self._pool_strides,
        )

        self._mlp_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization)

    def forward(self, *inputs):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.distributions.Categorical: Policy distribution.


        """
        cnn_output = self._cnn_module(*inputs)
        mlp_output = self._mlp_model(cnn_output)
        if not isinstance(mlp_output, Categorical):
            dist = Categorical(mlp_output)
        return dist
