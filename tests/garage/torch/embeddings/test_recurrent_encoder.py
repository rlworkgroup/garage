# pylint: disable=missing-docstring
"""This is a script to test the RecurrentEncoder module."""

import pickle

import pytest
import torch
import torch.nn as nn

from garage.torch.embeddings import RecurrentEncoder


class TestRecurrentEncoder:
    """Test for RecurrentEncoder."""
    # yapf: disable
    @pytest.mark.parametrize(
        'input_dim, output_dim, hidden_sizes, num_tasks, num_seq', [
            (1, 1, (1, ), 1, 3),
            (3, 3, (3, ), 1, 5),
            (5, 5, (5, 5), 2, 4),
            (7, 7, (7, 5, 7), 2, 5),
            (9, 9, (9, 7, 5, 9), 3, 10),
        ])
    # yapf: enable
    def test_module(self, input_dim, output_dim, hidden_sizes, num_tasks,
                    num_seq):
        """Test forward method."""
        input_val = torch.ones((num_tasks, num_seq, input_dim),
                               dtype=torch.float32)
        # last hidden size should match output size
        # output_dim is latent_dim
        module = RecurrentEncoder(input_dim=input_dim,
                                  output_dim=output_dim,
                                  hidden_nonlinearity=None,
                                  hidden_sizes=hidden_sizes,
                                  hidden_w_init=nn.init.ones_,
                                  output_w_init=nn.init.ones_)
        module.reset(num_tasks=num_tasks)
        output = module(input_val)

        # maps input of shape (task, seq, input_dim) to (task, 1, output_dim)
        expected_shape = [num_tasks, 1, output_dim]

        assert all([a == b for a, b in zip(output.shape, expected_shape)])

    # yapf: disable
    @pytest.mark.parametrize(
        'input_dim, output_dim, hidden_sizes, num_tasks, num_seq', [
            (1, 1, (1, ), 1, 3),
            (3, 3, (3, ), 1, 5),
            (5, 5, (5, 5), 2, 4),
            (7, 7, (7, 5, 7), 2, 5),
            (9, 9, (9, 7, 5, 9), 3, 10),
        ])
    # yapf: enable
    def test_is_pickleable(self, input_dim, output_dim, hidden_sizes,
                           num_tasks, num_seq):
        """Test is_pickeable."""
        input_val = torch.ones((num_tasks, num_seq, input_dim),
                               dtype=torch.float32)
        module = RecurrentEncoder(input_dim=input_dim,
                                  output_dim=output_dim,
                                  hidden_nonlinearity=None,
                                  hidden_sizes=hidden_sizes,
                                  hidden_w_init=nn.init.ones_,
                                  output_w_init=nn.init.ones_)
        module.reset(num_tasks=num_tasks)
        output1 = module(input_val)

        h = pickle.dumps(module)
        module_pickled = pickle.loads(h)
        module_pickled.reset(num_tasks=num_tasks)
        output2 = module_pickled(input_val)

        assert torch.all(torch.eq(output1, output2))
