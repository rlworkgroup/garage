"""Test CNN Module."""

import numpy as np
import pytest
import torch
from torch.autograd import Variable
import torch.nn as nn

from garage.torch.modules import CNNBaseModule


class TestCNNModule:
    """Test CNN Module."""

    def setup_method(self):
        self.input_dim = 5
        self.hidden_nonlinearity = torch.nn.ReLU

    @pytest.mark.parametrize('output_dim, kernel_sizes, hidden_sizes, strides',
                             [
                                 (1, (1, ), (32, ), (1, )),
                                 (1, (3, ), (32, ), (1, )),
                                 (2, (3, ), (32, ), (2, )),
                                 (2, (1, 1), (32, 64), (1, 1)),
                                 (3, (3, 3), (32, 64), (1, 1)),
                                 (3, (3, 3), (32, 64), (2, 2)),
                             ])
    def test_output_values(self, output_dim, kernel_sizes, strides,
                           hidden_sizes):
        """Test output values from CNNBaseModule.

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Ouput dimension.
            hidden_sizes (list[int]): Size of hidden layers.

        """
        dtype = torch.FloatTensor  # the CPU datatype
        x = torch.randn(output_dim, self.input_dim, 32, 32).type(dtype)
        input_val = Variable(x.type(dtype))
        # input_val = torch.ones([hidden_sizes, self.input_dim,
        # kernel_sizes, dtype=torch.float32)
        module_with_nonlinear_function_and_module = CNNBaseModule(
            input_dim=self.input_dim,
            output_dim=output_dim,
            kernel_sizes=kernel_sizes,
            strides=strides,
            hidden_nonlinearity=torch.relu,
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.xavier_uniform_,
            output_w_init=nn.init.xavier_uniform_,
            output_nonlinearity=nn.ReLU)

        module_with_nonlinear_module_instance_and_function = CNNBaseModule(
            input_dim=self.input_dim,
            output_dim=output_dim,
            kernel_sizes=kernel_sizes,
            strides=strides,
            hidden_nonlinearity=nn.ReLU(),
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_,
            output_nonlinearity=torch.relu)

        output1 = module_with_nonlinear_function_and_module(input_val)
        output2 = module_with_nonlinear_module_instance_and_function(input_val)

        expected_output = torch.full([1, output_dim],
                                     fill_value=5 * np.prod(hidden_sizes),
                                     dtype=torch.float32)

        assert torch.all(torch.eq(expected_output, output1))
        assert torch.all(torch.eq(expected_output, output2))

    def test_is_pickleable():
        pass

    def test_output_with_identity_filter():
        pass

    def test_output_with_random_filter():
        pass

    def test_output_with_max_pooling():
        pass

    def test_invalid_padding_mode(self):
        with pytest.raises(ValueError):
            self.cnn = CNNBaseModule(input_var=self.input_dim,
                                     filter_dims=(3, ),
                                     num_filters=(32, ),
                                     strides=(1, ),
                                     padding_mode='UNKNOWN')

    def test_invalid_padding_mode_max_pooling(self):
        with pytest.raises(ValueError):
            self.cnn = CNNBaseModule(input_var=self.input_dim,
                                     filter_dims=(3, ),
                                     num_filters=(32, ),
                                     strides=(1, ),
                                     max_pool=True,
                                     pool_shapes=(1, 1),
                                     pool_strides=(1, 1),
                                     padding_mode='UNKNOWN')
