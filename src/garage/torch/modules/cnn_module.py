"""CNN Module."""
import warnings

import akro
import numpy as np
import torch
from torch import nn

from garage import InOutSpec
from garage.torch import (expand_var, NonLinearity, output_height_2d,
                          output_width_2d)


# pytorch v1.6 issue, see https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class CNNModule(nn.Module):
    """Convolutional neural network (CNN) model in pytorch.

    Args:
        spec (garage.InOutSpec): Specification of inputs and outputs.
            The input should be in 'NCHW' format: [batch_size, channel, height,
            width]. Will print a warning if the channel size is not 1 or 3.
            If output_space is specified, then a final linear layer will be
            inserted to map to that dimensionality.
            If output_space is None, it will be filled in with the computed
            output space.
        image_format (str): Either 'NCHW' or 'NHWC'. Should match the input
            specification. Gym uses NHWC by default, but PyTorch uses NCHW by
            default.
        hidden_channels (tuple[int]): Number of output channels for CNN.
            For example, (3, 32) means there are two convolutional layers.
            The filter for the first conv layer outputs 3 channels
            and the second one outputs 32 channels.
        kernel_sizes (tuple[int]): Dimension of the conv filters.
            For example, (3, 5) means there are two convolutional layers.
            The filter for first layer is of dimension (3 x 3)
            and the second one is of dimension (5 x 5).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        paddings (tuple[int]): Amount of zero-padding added to both sides of
            the input of a conv layer.
        padding_mode (str): The type of padding algorithm to use, i.e.
            'constant', 'reflect', 'replicate' or 'circular' and
            by default is 'zeros'.
        hidden_nonlinearity (callable or torch.nn.Module):
            Activation function for intermediate dense layer(s).
            It should return a torch.Tensor. Set it to None to maintain a
            linear activation.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        max_pool (bool): Bool for using max-pooling or not.
        pool_shape (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all pooling layers are of the same
            shape (2, 2).
        pool_stride (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        layer_normalization (bool): Bool for using layer normalization or not.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        enable_cudnn_benchmarks (bool): Whether to enable cudnn benchmarks
            in `torch`. If enabled, the backend selects the CNN benchamark
            algorithm with the best performance.
    """

    def __init__(
            self,
            spec,
            image_format,
            hidden_channels,
            *,  # Many things after this are ints or tuples of ints.
            kernel_sizes,
            strides,
            paddings=0,
            padding_mode='zeros',
            hidden_nonlinearity=nn.ReLU,
            hidden_w_init=nn.init.xavier_uniform_,
            hidden_b_init=nn.init.zeros_,
            max_pool=False,
            pool_shape=None,
            pool_stride=1,
            layer_normalization=False,
            enable_cudnn_benchmarks=True):
        super().__init__()
        assert len(hidden_channels) > 0
        # PyTorch forces us to use NCHW internally.
        in_channels, height, width = _check_spec(spec, image_format)
        self._format = image_format
        kernel_sizes = expand_var('kernel_sizes', kernel_sizes,
                                  len(hidden_channels), 'hidden_channels')
        strides = expand_var('strides', strides, len(hidden_channels),
                             'hidden_channels')
        paddings = expand_var('paddings', paddings, len(hidden_channels),
                              'hidden_channels')
        pool_shape = expand_var('pool_shape', pool_shape, len(hidden_channels),
                                'hidden_channels')
        pool_stride = expand_var('pool_stride', pool_stride,
                                 len(hidden_channels), 'hidden_channels')

        self._cnn_layers = nn.Sequential()
        torch.backends.cudnn.benchmark = enable_cudnn_benchmarks

        # In case there are no hidden channels, handle output case.
        out_channels = in_channels
        for i, out_channels in enumerate(hidden_channels):
            conv_layer = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_sizes[i],
                                   stride=strides[i],
                                   padding=paddings[i],
                                   padding_mode=padding_mode)
            height = output_height_2d(conv_layer, height)
            width = output_width_2d(conv_layer, width)
            hidden_w_init(conv_layer.weight)
            hidden_b_init(conv_layer.bias)
            self._cnn_layers.add_module(f'conv_{i}', conv_layer)

            if layer_normalization:
                self._cnn_layers.add_module(
                    f'layer_norm_{i}',
                    nn.LayerNorm((out_channels, height, width)))

            if hidden_nonlinearity:
                self._cnn_layers.add_module(f'non_linearity_{i}',
                                            NonLinearity(hidden_nonlinearity))

            if max_pool:
                pool = nn.MaxPool2d(kernel_size=pool_shape[i],
                                    stride=pool_stride[i])
                height = output_height_2d(pool, height)
                width = output_width_2d(pool, width)
                self._cnn_layers.add_module(f'max_pooling_{i}', pool)

            in_channels = out_channels

        output_dims = out_channels * height * width

        if spec.output_space is None:
            final_spec = InOutSpec(
                spec.input_space,
                akro.Box(low=-np.inf, high=np.inf, shape=(output_dims, )))
            self._final_layer = None
        else:
            final_spec = spec
            # Checked at start of __init__
            self._final_layer = nn.Linear(output_dims,
                                          spec.output_space.shape[0])

        self.spec = final_spec

    # pylint: disable=arguments-differ
    def forward(self, x):
        """Forward method.

        Args:
            x (torch.Tensor): Input values. Should match image_format
                specified at construction (either NCHW or NCWH).

        Returns:
            List[torch.Tensor]: Output values

        """
        # Transform single values into batch, if necessary.
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # This should be the single place in torch that image normalization
        # happens
        if isinstance(self.spec.input_space, akro.Image):
            x = torch.div(x, 255.0)
        assert len(x.shape) == 4
        if self._format == 'NHWC':
            # Convert to internal NCHW format
            x = x.permute((0, 3, 1, 2))
        for layer in self._cnn_layers:
            x = layer(x)
        if self._format == 'NHWC':
            # Convert back to NHWC (just in case)
            x = x.permute((0, 2, 3, 1))
        # Remove non-batch dimensions
        x = x.reshape(x.shape[0], -1)
        # Apply final linearity, if it was requested.
        if self._final_layer is not None:
            x = self._final_layer(x)
        return x


def _check_spec(spec, image_format):
    """Check that an InOutSpec is suitable for a CNNModule.

    Args:
        spec (garage.InOutSpec): Specification of inputs and outputs.  The
            input should be in 'NCHW' format: [batch_size, channel, height,
            width].  Will print a warning if the channel size is not 1 or 3.
            If output_space is specified, then a final linear layer will be
            inserted to map to that dimensionality.  If output_space is None,
            it will be filled in with the computed output space.
        image_format (str): Either 'NCHW' or 'NHWC'. Should match the input
            specification. Gym uses NHWC by default, but PyTorch uses NCHW by
            default.

    Returns:
        tuple[int, int, int]: The input channels, height, and width.

    Raises:
        ValueError: If spec isn't suitable for a CNNModule.

    """
    # pylint: disable=no-else-raise
    input_space = spec.input_space
    output_space = spec.output_space
    # Don't use isinstance, since akro.Space is guaranteed to inherit from
    # gym.Space
    if getattr(input_space, 'shape', None) is None:
        raise ValueError(
            f'input_space to CNNModule is {input_space}, but should be an '
            f'akro.Box or akro.Image')
    elif len(input_space.shape) != 3:
        raise ValueError(
            f'Input to CNNModule is {input_space}, but should have three '
            f'dimensions.')
    if (output_space is not None and not (hasattr(output_space, 'shape')
                                          and len(output_space.shape) == 1)):
        raise ValueError(
            f'output_space to CNNModule is {output_space}, but should be '
            f'an akro.Box with a single dimension or None')
    if image_format == 'NCHW':
        in_channels = spec.input_space.shape[0]
        height = spec.input_space.shape[1]
        width = spec.input_space.shape[2]
    elif image_format == 'NHWC':
        height = spec.input_space.shape[0]
        width = spec.input_space.shape[1]
        in_channels = spec.input_space.shape[2]
    else:
        raise ValueError(
            f'image_format has value {image_format!r}, but must be either '
            f"'NCHW' or 'NHWC'")
    if in_channels not in (1, 3):
        warnings.warn(
            f'CNNModule input has {in_channels} channels, but '
            f'1 or 3 channels are typical. Consider changing the CNN '
            f'image_format.')
    return in_channels, height, width
