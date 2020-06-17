"""CNN Module."""
import copy

from torch import nn

# from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule


class CNNBaseModule(nn.Module):
    """Convolutional neural network (CNN) module in pytorch.

    Note:
        Based on 'NCHW' data format: [batch_size, channel, height, width].

    A PyTorch module composed only of a CNN with
    multiple parallel output layers which maps real-valued inputs to
    real-valued outputs. The length of outputs is n_heads and shape of each
    output element is depend on each output dimension

    Args:
        input_dim (int): Dimension of the network input.
        output_dim (int): Dimension of the network output.
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
        hidden_nonlinearity (callable or torch.nn.Module):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
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
        output_nonlinearity (callable or torch.nn.Module):
            Activation function for output dense layer. It should return a
            torch.Tensor. Set it to None to maintain a linear activation.
            Size of the parameter should be 1 or equal to n_head
        output_w_init (callable): Initializer function for
            the weight of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        output_b_init (callable): Initializer function for
            the bias of output dense layer(s). The function should return a
            torch.Tensor. Size of the parameter should be 1 or equal to n_head
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(
            self,
            input_dim,
            output_dim,
            kernel_sizes,
            strides,
            hidden_sizes,
            hidden_nonlinearity=nn.ReLU,
            hidden_w_init=nn.init.xavier_uniform_,
            hidden_b_init=nn.init.zeros_,
            paddings=[
                0,
            ],
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

        self._layers = nn.ModuleList()
        prev_size = input_dim
        for index, (size, kernel_size, stride, padding) in enumerate(
                zip(hidden_sizes, kernel_sizes, strides, paddings)):
            hidden_layers = nn.Sequential()
            # conv 2D layer
            conv_layer = _conv(in_channels=prev_size,
                               out_channels=size,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               padding_mode=padding_mode)
            hidden_w_init(conv_layer.weight)
            hidden_b_init(conv_layer.bias)
            hidden_layers.add_module('conv_{}'.format(index), conv_layer)

            # layer normalization
            if layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                         nn.LayerNorm(size))

            # non-linear function
            if hidden_nonlinearity:
                hidden_layers.add_module('non_linearity',
                                         _NonLinearity(hidden_nonlinearity))
            # max-pooling
            if max_pool:
                hidden_layers.add_module(
                    'max_pooling',
                    nn.MaxPool2d(kernel_size=pool_shapes, stride=pool_strides))

            self._layers.append(hidden_layers)
            prev_size = size

        # flattening
        self._layers.append(_Flatten())

        self._output_layers = nn.ModuleList()
        output_layer = nn.Sequential()
        linear_layer = nn.Linear(prev_size, output_dim)
        output_w_init(linear_layer.weight)
        output_b_init(linear_layer.bias)
        output_layer.add_module('linear', linear_layer)
        if output_nonlinearity:
            output_layer.add_module('non_linearity',
                                    _NonLinearity(output_nonlinearity))
        self._output_layers.append(output_layer)

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, *, input_dim)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        x = input_val
        for layer in self._layers:
            x = layer(x)

        return [output_layer(x) for output_layer in self._output_layers]


def _conv(in_channels,
          out_channels,
          kernel_size,
          stride=1,
          padding=0,
          padding_mode='zeros',
          dilation=1,
          bias=True):
    """Helper function for performing convolution.

    Args:
        in_channels (int):
            Number of channels in the input image
        out_channels (int):
            Number of channels produced by the convolution
        kernel_size (int or tuple):
            Size of the convolving kernel
        stride (int or tuple): Stride of the convolution.
            Default: 1
        padding (int or tuple): Zero-padding added to both sides
            of the input. Default: 0
        padding_mode (string): 'zeros', 'reflect', 'replicate'
            or 'circular'. Default: 'zeros'
        dilation (int or tuple): Spacing between kernel elements.
            Default: 1
        bias (bool): If True, adds a learnable bias to the output.
            Default: True

    Returns:
        torch.Tensor: The output of the 2D convolution.

    """
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     padding_mode=padding_mode,
                     dilation=dilation,
                     bias=bias)


class _Flatten(nn.Module):
    # pylint: disable=arguments-differ, missing-return-type-doc
    def forward(self, x):
        """Collapse the C x H x W values per representation into a single long vector.

        Args:
            x (torch.tensor): batch of data

        Returns:
            View of that data (analogous to numpy.reshape)

        """
        # x.shape read in [N, C, H, W]
        # "flatten" the C * H * W values into a single vector per data, such that the shape becomes: [batch_size, C x H x W]
        return x.view(x.shape[0], -1)


class _NonLinearity(nn.Module):
    """Wrapper class for non linear function or module.

    Args:
        non_linear (callable or type): Non-linear function or type to be
            wrapped.

    """

    def __init__(self, non_linear):
        super().__init__()

        if isinstance(non_linear, type):
            self.module = non_linear()
        elif callable(non_linear):
            self.module = copy.deepcopy(non_linear)
        else:
            raise ValueError(
                'Non linear function {} is not supported'.format(non_linear))

    # pylint: disable=arguments-differ
    def forward(self, input_value):
        """Forward method.

        Args:
            input_value (torch.Tensor): Input values

        Returns:
            torch.Tensor: Output value

        """
        return self.module(input_value)

    # pylint: disable=missing-return-doc, missing-return-type-doc
    def __repr__(self):
        return repr(self.module)
