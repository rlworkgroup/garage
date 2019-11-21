# pylint: disable=missing-docstring
"""This is a script to test the RecurrentEncoder module."""

import pickle

import pytest
import torch
import torch.nn as nn

from garage.torch.algos import RecurrentEncoder


class TestRecurrentEncoder:
    """Test for RecurrentEncoder."""
    # yapf: disable
    @pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', [
        (1, 1, (1, )),
        (3, 3, (3, )),
        (5, 5, (5, 5)),
        (7, 7, (7, 5, 7)),
        (9, 9, (9, 7, 5, 9)),
    ])
    # yapf: enable
    def test_module(self, input_dim, output_dim, hidden_sizes):
        """Test forward method."""
        input_val = torch.ones((input_dim, input_dim, input_dim),
                               dtype=torch.float32)
        # last hidden size should match output size
        module = RecurrentEncoder(input_dim=input_dim,
                                  output_dim=output_dim,
                                  hidden_nonlinearity=None,
                                  hidden_sizes=hidden_sizes,
                                  hidden_w_init=nn.init.ones_,
                                  output_w_init=nn.init.ones_)
        module.reset(num_tasks=input_dim)
        output = module(input_val)

        expected_shape = torch.Tensor([input_dim, input_dim])

        assert torch.all(torch.eq(torch.Tensor((output.shape)),
                                  expected_shape))

    # yapf: disable
    @pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', [
        (1, 1, (1, )),
        (3, 3, (3, )),
        (5, 5, (5, 5)),
        (7, 7, (7, 5, 7)),
        (9, 9, (9, 7, 5, 9)),
    ])
    # yapf: enable
    def test_is_pickleable(self, input_dim, output_dim, hidden_sizes):
        """Test is_pickeable."""
        input_val = torch.ones((input_dim, input_dim, input_dim),
                               dtype=torch.float32)
        module = RecurrentEncoder(input_dim=input_dim,
                                  output_dim=output_dim,
                                  hidden_nonlinearity=None,
                                  hidden_sizes=hidden_sizes,
                                  hidden_w_init=nn.init.ones_,
                                  output_w_init=nn.init.ones_)
        module.reset(num_tasks=input_dim)
        output1 = module(input_val)

        h = pickle.dumps(module)
        module_pickled = pickle.loads(h)
        module_pickled.reset(num_tasks=input_dim)
        output2 = module_pickled(input_val)

        assert torch.all(torch.eq(output1, output2))
