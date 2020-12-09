"""Test CategoricalGRUModule."""
import pickle

import numpy as np
import pytest
import torch
from torch.distributions import Categorical
import torch.nn as nn

from garage.torch.modules.categorical_gru_module import CategoricalGRUModule


class TestCategoricalGRUModule:
    def setup_method(self):
        self.batch_size = 1
        self.time_step = 1
        self.feature_shape = 2
        self.output_dim = 1
        self.dtype = torch.float32

        self.input = torch.full(
            (self.batch_size, self.time_step, self.feature_shape), 1.)
    
    def test_dist(self):
        model = CategoricalGRUModule(input_dim=self.feature_shape, output_dim=self.output_dim, hidden_dim=1)
        dist = model(self.input)
        assert isinstance(dist, Categorical)
    
    @pytest.mark.parametrize('output_dim', [1, 2, 5, 10])
    def test_output_normalized(self, output_dim):
        model = CategoricalGRUModule(input_dim=self.feature_shape, output_dim=output_dim, hidden_dim=1)
        dist = model(self.input)
        return np.isclose(dist.probs.squeeze().sum().detach().numpy(), 1)