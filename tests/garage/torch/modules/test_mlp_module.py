import pickle
import unittest

from nose2.tools.params import params
import numpy as np
import torch
import torch.nn as nn

from garage.torch.modules import MLPModule


class TestMLPModel(unittest.TestCase):
    @params((5, 1, (1, )), (5, 1, (2, )), (5, 2, (3, )), (5, 2, (1, 1)),
            (5, 3, (2, 2)))
    def test_output_values(self, input_dim, output_dim, hidden_sizes):
        input_val = torch.ones([1, 5], dtype=torch.float32)
        module = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_)
        output = module(input_val)

        expected_output = torch.full([1, output_dim],
                                     fill_value=5 * np.prod(hidden_sizes),
                                     dtype=torch.float32)

        self.assertEqual(torch.all(torch.eq(output, expected_output)), True)

    @params((5, 1, (1, )), (5, 1, (2, )), (5, 2, (3, )), (5, 2, (1, 1)),
            (5, 3, (2, 2)))
    def test_is_pickleable(self, input_dim, output_dim, hidden_sizes):
        input_val = torch.ones([1, 5], dtype=torch.float32)
        module = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_)
        output1 = module(input_val)

        h = pickle.dumps(module)
        model_pickled = pickle.loads(h)
        output2 = model_pickled(input_val)

        assert np.array_equal(torch.all(torch.eq(output1, output2)), True)
