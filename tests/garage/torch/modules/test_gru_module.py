"""Test GRUModule."""
import pickle
import numpy as np

import torch

from garage.torch.modules import GRUModule


class TestGRUModule:
    """Test GRUModule."""

    def setup_method(self):
        self.input_dim = 28
        self.hidden_dim = 128
        self.layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
        self.output_dim = 10

        self.batch_size = 100
        self.input = torch.zeros(
            (self.batch_size, self.input_dim, self.input_dim))

    def test_output_values(self):
        model = GRUModule(self.input_dim, self.hidden_dim, self.layer_dim,
                          self.output_dim)

        outputs, output, _, _ = model(self.input)  # read step output
        assert output.size() == (self.batch_size, self.output_dim
                                 )  # (batch_size, output_dim)
        assert outputs.shape == (self.input_dim, self.batch_size,
                                 self.output_dim
                                 )  # (input_dim, batch_size, output_dim)

    def test_is_pickleable(self):
        model = GRUModule(self.input_dim, self.hidden_dim, self.layer_dim,
                          self.output_dim)
        outputs1, output1, _, _ = model(self.input)

        h = pickle.dumps(model)
        model_pickled = pickle.loads(h)
        outputs2, output2, _, _ = model_pickled(self.input)

        assert np.array_equal(torch.all(torch.eq(outputs1, outputs2)), True)
        assert np.array_equal(torch.all(torch.eq(output1, output2)), True)
