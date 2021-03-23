"""Test CNNModule."""
import pickle

import akro
import numpy as np
import pytest
import torch
import torch.nn as nn

from garage import InOutSpec
from garage.torch.modules import CNNModule
from garage.torch.modules.cnn_module import _check_spec


class TestCNNModule:
    """Test CNNModule."""

    def setup_method(self):
        self.batch_size = 64
        self.input_width = 32
        self.input_height = 32
        self.in_channel = 3
        self.dtype = torch.float32
        self.input_spec = InOutSpec(
            akro.Box(
                shape=[self.in_channel, self.input_height, self.input_width],
                high=np.inf,
                low=np.inf), None)
        self.input = torch.zeros(
            (self.batch_size, self.in_channel, self.input_height,
             self.input_width),
            dtype=self.dtype)  # minibatch size 64, image size [3, 32, 32]

    @pytest.mark.parametrize(
        'kernel_sizes, hidden_channels, strides, paddings', [
            ((1, ), (32, ), (1, ), (0, )),
            ((3, ), (32, ), (1, ), (0, )),
            ((3, ), (32, ), (2, ), (0, )),
            ((5, ), (12, ), (1, ), (2, )),
            ((1, 1), (32, 64), (1, 1), (0, 0)),
            ((3, 3), (32, 64), (1, 1), (0, 0)),
            ((3, 3), (32, 64), (2, 2), (0, 0)),
        ])
    def test_output_values(self, kernel_sizes, hidden_channels, strides,
                           paddings):
        """Test output values from CNNBaseModule.

        Args:
            kernel_sizes (tuple[int]): Kernel sizes.
            hidden_channels (tuple[int]): hidden channels.
            strides (tuple[int]): strides.
            paddings (tuple[int]): value of zero-padding.

        """
        module_with_nonlinear_function_and_module = CNNModule(
            self.input_spec,
            image_format='NCHW',
            hidden_channels=hidden_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            padding_mode='zeros',
            hidden_nonlinearity=torch.relu,
            hidden_w_init=nn.init.xavier_uniform_)

        module_with_nonlinear_module_instance_and_function = CNNModule(
            self.input_spec,
            image_format='NCHW',
            hidden_channels=hidden_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            padding_mode='zeros',
            hidden_nonlinearity=nn.ReLU(),
            hidden_w_init=nn.init.xavier_uniform_)

        output1 = module_with_nonlinear_function_and_module(self.input)
        output2 = module_with_nonlinear_module_instance_and_function(
            self.input)

        current_size = self.input_width
        for (filter_size, stride, padding) in zip(kernel_sizes, strides,
                                                  paddings):
            # padding = float((filter_size - 1) / 2) # P = (F - 1) /2
            current_size = int(
                (current_size - filter_size + padding * 2) /
                stride) + 1  # conv formula = ((W - F + 2P) / S) + 1
        flatten_shape = current_size * current_size * hidden_channels[-1]

        expected_output = torch.zeros((self.batch_size, flatten_shape))

        assert np.array_equal(torch.all(torch.eq(output1, expected_output)),
                              True)
        assert np.array_equal(torch.all(torch.eq(output2, expected_output)),
                              True)

    @pytest.mark.parametrize(
        'kernel_sizes, hidden_channels, strides, paddings', [
            ((5, 3), (12, 8), (1, 1), (2, 1)),
        ])
    def test_output_values_with_unequal_stride_with_padding(
            self, hidden_channels, kernel_sizes, strides, paddings):
        """Test output values with unequal stride and padding from CNNModule.

        Args:
            kernel_sizes (tuple[int]): Kernel sizes.
            hidden_channels (tuple[int]): hidden channels.
            strides (tuple[int]): strides.
            paddings (tuple[int]): value of zero-padding.

        """
        model = CNNModule(self.input_spec,
                          image_format='NCHW',
                          hidden_channels=hidden_channels,
                          kernel_sizes=kernel_sizes,
                          strides=strides,
                          paddings=paddings,
                          padding_mode='zeros',
                          hidden_nonlinearity=torch.relu,
                          hidden_w_init=nn.init.xavier_uniform_)
        output = model(self.input)

        current_size = self.input_width
        for (filter_size, stride, padding) in zip(kernel_sizes, strides,
                                                  paddings):
            # padding = float((filter_size - 1) / 2) # P = (F - 1) /2
            current_size = int(
                (current_size - filter_size + padding * 2) /
                stride) + 1  # conv formula = ((W - F + 2P) / S) + 1
        flatten_shape = current_size * current_size * hidden_channels[-1]

        expected_output = torch.zeros((self.batch_size, flatten_shape))
        assert np.array_equal(torch.all(torch.eq(output, expected_output)),
                              True)

    @pytest.mark.parametrize('hidden_channels, kernel_sizes, strides',
                             [((32, ), (1, ), (1, )), ((32, ), (3, ), (1, )),
                              ((32, ), (3, ), (1, )),
                              ((32, 64), (1, 1), (1, 1)),
                              ((32, 64), (3, 3), (1, 1))])
    def test_is_pickleable(self, hidden_channels, kernel_sizes, strides):
        """Check CNNModule is pickeable.

        Args:
            hidden_channels (tuple[int]): hidden channels.
            kernel_sizes (tuple[int]): Kernel sizes.
            strides (tuple[int]): strides.

        """
        model = CNNModule(self.input_spec,
                          image_format='NCHW',
                          hidden_channels=hidden_channels,
                          kernel_sizes=kernel_sizes,
                          strides=strides)
        output1 = model(self.input)

        h = pickle.dumps(model)
        model_pickled = pickle.loads(h)
        output2 = model_pickled(self.input)

        assert np.array_equal(torch.all(torch.eq(output1, output2)), True)

    @pytest.mark.parametrize('kernel_sizes, hidden_channels, '
                             'strides, pool_shape, pool_stride',
                             [((1, ), (32, ), (1, ), 1, 1),
                              ((3, ), (32, ), (1, ), 1, 1),
                              ((3, ), (32, ), (2, ), 2, 2),
                              ((1, 1), (32, 64), (1, 1), 1, 1),
                              ((3, 3), (32, 64), (1, 1), 1, 1),
                              ((3, 3), (32, 64), (1, 1), 2, 2)])
    def test_output_with_max_pooling(self, kernel_sizes, hidden_channels,
                                     strides, pool_shape, pool_stride):
        model = CNNModule(self.input_spec,
                          image_format='NCHW',
                          hidden_channels=hidden_channels,
                          kernel_sizes=kernel_sizes,
                          strides=strides,
                          max_pool=True,
                          pool_shape=[(pool_shape, pool_shape)],
                          pool_stride=[(pool_stride, pool_stride)])
        x = model(self.input)
        fc_w = torch.zeros((x.shape[1], 10))
        fc_b = torch.zeros(10)
        result = x.mm(fc_w) + fc_b
        assert result.size() == torch.Size([64, 10])

    @pytest.mark.parametrize('hidden_nonlinear', [('test'), (object())])
    def test_no_head_invalid_settings(self, hidden_nonlinear):
        """Check CNNModule throws exception with invalid non-linear functions.

        Args:
            hidden_nonlinear (callable or torch.nn.Module): Non-linear
                functions for hidden layers.

        """
        expected_msg = 'Non linear function .* is not supported'
        with pytest.raises(ValueError, match=expected_msg):
            CNNModule(self.input_spec,
                      image_format='NCHW',
                      hidden_channels=(32, ),
                      kernel_sizes=(3, ),
                      strides=(1, ),
                      hidden_nonlinearity=hidden_nonlinear)


@pytest.mark.parametrize('kernel_sizes, hidden_channels, '
                         'strides, pool_shape, pool_stride',
                         [((1, ), (32, ), (1, ), 1, 1),
                          ((3, ), (32, ), (1, ), 1, 1),
                          ((3, ), (32, ), (2, ), 2, 2),
                          ((1, 1), (32, 64), (1, 1), 1, 1),
                          ((3, 2), (32, 64), (1, 1), 1, 1),
                          ((3, 2), (32, 64), (2, 1), 1, 1),
                          ((3, 3), (32, 64), (1, 1), 1, 1),
                          ((3, 3), (32, 64), (1, 1), 2, 2)])
def test_set_output_size(kernel_sizes, hidden_channels, strides, pool_shape,
                         pool_stride):
    spec = InOutSpec(akro.Box(shape=[3, 19, 15], high=np.inf, low=-np.inf),
                     akro.Box(shape=[200], high=np.inf, low=-np.inf))
    model = CNNModule(spec,
                      image_format='NCHW',
                      hidden_channels=hidden_channels,
                      kernel_sizes=kernel_sizes,
                      strides=strides,
                      pool_shape=[(pool_shape, pool_shape)],
                      pool_stride=[(pool_stride, pool_stride)],
                      layer_normalization=True)
    images = torch.ones(10, 3, 19, 15)
    x = model(images)
    assert x.shape == (10, 200)


def test_check_spec():
    with pytest.raises(ValueError, match='should be an akro.Box'):
        # Input space is not Box or Image
        _check_spec(InOutSpec(akro.Dict(), None), 'NCHW')
    with pytest.raises(ValueError, match='should have three dimensions'):
        # Too many input dimensions
        _check_spec(
            InOutSpec(akro.Box(shape=[1, 1, 1, 1], low=-np.inf, high=np.inf),
                      None), 'NCHW')
    with pytest.raises(ValueError, match='akro.Box with a single dimension'):
        # Output is not one-dimensional
        _check_spec(
            InOutSpec(akro.Box(shape=[1, 1, 1], low=-np.inf, high=np.inf),
                      akro.Box(
                          shape=[1, 1],
                          low=-np.inf,
                          high=np.inf,
                      )), 'NCHW')
    with pytest.warns(UserWarning):
        # 4 color channels should warn
        _check_spec(
            InOutSpec(akro.Box(shape=[4, 1, 1], low=-np.inf, high=np.inf),
                      None), 'NCHW')
