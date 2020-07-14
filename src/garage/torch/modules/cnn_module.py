"""CNN Module."""
import copy

from torch import nn

from garage.torch import flatten_to_single_vector, NonLinearity


# pylint: disable=unused-argument
class CNNModule(nn.Module):
    """Convolutional neural network (CNN) model in pytorch.

    Args:
        input_var (pytorch.tensor): Input tensor of the model.
            Based on 'NCHW' data format: [batch_size, channel, height, width].
        kernel_sizes (tuple[int]): Dimension of the conv filters.
            For example, (3, 5) means there are two convolutional layers.
            The filter for first layer is of dimension (3 x 3)
            and the second one is of dimension (5 x 5).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_channels (tuple[int]): Number of output channels for CNN.
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
        paddings (tuple[int]): Amount of zero-padding added to both sides of
            the input of a conv layer.
        padding_mode (str): The type of padding algorithm to use, i.e.
            'constant', 'reflect', 'replicate' or 'circular' and
            by default is 'zeros'.
        max_pool (bool): Bool for using max-pooling or not.
        pool_shape (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all pooling layers are of the same
            shape (2, 2).
        pool_stride (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        layer_normalization (bool): Bool for using layer normalization or not.
        n_layers (int): number of convolutional layer.
        is_image (bool): Whether observations are images or not.
    """

    def __init__(self,
                 input_var,
                 hidden_channels,
                 kernel_sizes,
                 strides=1,
                 hidden_nonlinearity=nn.ReLU,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 paddings=0,
                 padding_mode='zeros',
                 max_pool=False,
                 pool_shape=None,
                 pool_stride=1,
                 layer_normalization=False,
                 n_layers=None,
                 is_image=True):
        super().__init__()
        self._hidden_channels = hidden_channels
        self._kernel_sizes = kernel_sizes
        self._strides = strides
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._paddings = paddings
        self._padding_mode = padding_mode
        self._max_pool = max_pool
        self._pool_shape = pool_shape
        self._pool_stride = pool_stride
        self._layer_normalization = layer_normalization
        self._is_image = is_image

        self._cnn_layers = nn.ModuleList()
        self._in_channel = input_var.shape[1]  # read in N, C, H, W
        self._CNNCell()

    @classmethod
    def _check_parameter_for_output_layer(cls, var, n_layers):
        """Check input parameters for conv layer are valid.

        Args:
            var (any): variable to be checked
            n_layers (int): number of layers

        Returns:
            list: list of variables (length of n_layers)

        Raises:
            ValueError: if the variable is a list but length of the variable
                is not equal to n_layers

        """
        if isinstance(var, (list, tuple)):
            if len(var) == 1:
                return list(var) * n_layers
            if len(var) == n_layers:
                return var
            msg = ('{} should be either an integer or a collection of length '
                   'n_layers ({}), but got {} instead.')
            raise ValueError(msg.format(str(var), n_layers, var))
        return [copy.deepcopy(var) for _ in range(n_layers)]

    # pylint: disable=arguments-differ
    def forward(self, input_val):
        """Forward method.

        Args:
            input_val (torch.Tensor): Input values with (N, C, H, W)
                shape.

        Returns:
            List[torch.Tensor]: Output values

        """
        if self._is_image:
            input_val /= 255.0
        x = input_val
        for layer in self._cnn_layers:
            x = layer(x)
        x = flatten_to_single_vector(x)
        return x

    def _CNNCell(self):
        """Helper function for initializing convolutional layer(s)."""
        prev_channel = self._in_channel
        for index, (channel, kernel_size, stride) in enumerate(
                zip(self._hidden_channels, self._kernel_sizes, self._strides)):
            hidden_layers = nn.Sequential()

            if isinstance(self._paddings, (list, tuple)):
                padding = self._paddings[index]
            elif isinstance(self._paddings, int):
                padding = self._paddings

            # conv 2D layer
            conv_layer = _conv(in_channels=prev_channel,
                               out_channels=channel,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
            self._hidden_w_init(conv_layer.weight)
            self._hidden_b_init(conv_layer.bias)
            hidden_layers.add_module('conv_{}'.format(index), conv_layer)

            # layer normalization
            if self._layer_normalization:
                hidden_layers.add_module('layer_normalization',
                                         nn.LayerNorm(channel))

            # non-linear function
            if self._hidden_nonlinearity:
                hidden_layers.add_module(
                    'non_linearity', NonLinearity(self._hidden_nonlinearity))

            # max-pooling
            if self._max_pool:
                hidden_layers.add_module(
                    'max_pooling',
                    nn.MaxPool2d(kernel_size=self._pool_shape,
                                 stride=self._pool_stride))

            self._cnn_layers.append(hidden_layers)
            prev_channel = channel


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
