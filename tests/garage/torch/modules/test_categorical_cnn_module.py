"""Test CategoricalCNNModule."""
import pickle

import numpy as np
import pytest
import torch
from torch.distributions import Categorical
import torch.nn as nn

from garage.torch.modules.categorical_cnn_module import CategoricalCNNModule


class TestCategoricalCNNModule:
    """Test CategoricalCNNModule."""

    def setup_method(self):
        self.batch_size = 64
        self.input_width = 32
        self.input_height = 32
        self.in_channel = 3
        self.dtype = torch.float32
        self.input = torch.zeros(
            (self.batch_size, self.in_channel, self.input_height,
             self.input_width),
            dtype=self.dtype)  # minibatch size 64, image size [3, 32, 32]

    def test_dist(self):
        model = CategoricalCNNModule(
            input_var=self.input,
            output_dim=1,
            kernel_sizes=((3), ),
            hidden_channels=((5), ),
            strides=(1, ),
        )
        dist = model(self.input)
        assert isinstance(dist, Categorical)

    @pytest.mark.parametrize(
        'output_dim, hidden_channels, kernel_sizes, strides, hidden_sizes', [
            (1, (1, ), (1, ), (1, ), (1, )),
            (1, (3, ), (3, ), (2, ), (2, )),
            (1, (3, ), (3, ), (2, ), (3, )),
            (2, (3, 3), (3, 3), (2, 2), (1, 1)),
            (3, (3, 3), (3, 3), (2, 2), (2, 2)),
        ])
    def test_is_pickleable(self, output_dim, hidden_channels, kernel_sizes,
                           strides, hidden_sizes):
        model = CategoricalCNNModule(input_var=self.input,
                                     output_dim=output_dim,
                                     kernel_sizes=kernel_sizes,
                                     hidden_channels=hidden_channels,
                                     strides=strides,
                                     hidden_sizes=hidden_sizes,
                                     hidden_nonlinearity=None,
                                     hidden_w_init=nn.init.xavier_uniform_,
                                     output_w_init=nn.init.zeros_)
        dist1 = model(self.input)

        h = pickle.dumps(model)
        model_pickled = pickle.loads(h)
        dist2 = model_pickled(self.input)

        assert np.array_equal(dist1.probs.shape, dist2.probs.shape)
        assert np.array_equal(torch.all(torch.eq(dist1.probs, dist2.probs)),
                              True)
