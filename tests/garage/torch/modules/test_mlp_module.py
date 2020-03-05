"""Test MLPModule."""

import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn

from garage.torch.modules import MLPModule


class TestMLPModel:
    """Test MLPModule."""
    # yapf: disable
    @pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', [
        (5, 1, (1, )),
        (5, 1, (2, )),
        (5, 2, (3, )),
        (5, 2, (1, 1)),
        (5, 3, (2, 2)),
    ])
    # yapf: enable
    def test_output_values(self, input_dim, output_dim, hidden_sizes):
        """Test output values from MLPModule.

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Ouput dimension.
            hidden_sizes (list[int]): Size of hidden layers.

        """
        input_val = torch.ones([1, input_dim], dtype=torch.float32)
        module_with_nonlinear_function_and_module = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_nonlinearity=torch.relu,
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_,
            output_nonlinearity=torch.nn.ReLU)

        module_with_nonlinear_module_instance_and_function = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_nonlinearity=torch.nn.ReLU(),
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

    # yapf: disable
    @pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', [
        (5, 1, (1, )),
        (5, 1, (2, )),
        (5, 2, (3, )),
        (5, 2, (1, 1)),
        (5, 3, (2, 2)),
    ])
    # yapf: enable
    def test_is_pickleable(self, input_dim, output_dim, hidden_sizes):
        """Check MLPModule is pickeable.

        Args:
            input_dim (int): Input dimension.
            output_dim (int): Ouput dimension.
            hidden_sizes (list[int]): Size of hidden layers.

        """
        input_val = torch.ones([1, input_dim], dtype=torch.float32)
        module = MLPModule(input_dim=input_dim,
                           output_dim=output_dim,
                           hidden_nonlinearity=torch.relu,
                           hidden_sizes=hidden_sizes,
                           hidden_w_init=nn.init.ones_,
                           output_w_init=nn.init.ones_,
                           output_nonlinearity=torch.nn.ReLU)

        output1 = module(input_val)

        h = pickle.dumps(module)
        model_pickled = pickle.loads(h)
        output2 = model_pickled(input_val)

        assert np.array_equal(torch.all(torch.eq(output1, output2)), True)

    # yapf: disable
    @pytest.mark.parametrize('hidden_nonlinear, output_nonlinear', [
        (torch.nn.ReLU, 'test'),
        ('test', torch.relu),
        (object(), torch.tanh),
        (torch.tanh, object())
    ])
    # yapf: enable
    def test_no_head_invalid_settings(self, hidden_nonlinear,
                                      output_nonlinear):
        """Check MLPModule throws exception with invalid non-linear functions.

        Args:
            hidden_nonlinear (callable or torch.nn.Module): Non-linear
                functions for hidden layers.
            output_nonlinear (callable or torch.nn.Module): Non-linear
                functions for output layer.

        """
        expected_msg = 'Non linear function .* is not supported'
        with pytest.raises(ValueError, match=expected_msg):
            MLPModule(input_dim=3,
                      output_dim=5,
                      hidden_sizes=(2, 3),
                      hidden_nonlinearity=hidden_nonlinear,
                      output_nonlinearity=output_nonlinear)

    def test_mlp_with_learnable_non_linear_function(self):
        """Test MLPModule with learnable non-linear functions."""
        input_dim, output_dim, hidden_sizes = 1, 1, (3, 2)

        input_val = -torch.ones([1, input_dim], dtype=torch.float32)
        module = MLPModule(input_dim=input_dim,
                           output_dim=output_dim,
                           hidden_nonlinearity=torch.nn.PReLU(init=10.),
                           hidden_sizes=hidden_sizes,
                           hidden_w_init=nn.init.ones_,
                           output_w_init=nn.init.ones_,
                           output_nonlinearity=torch.nn.PReLU(init=1.))

        output = module(input_val)
        output.sum().backward()

        for tt in module.parameters():
            assert torch.all(torch.ne(tt.grad, 0))
