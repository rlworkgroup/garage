"""Test GRUModule."""
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

        output = model(self.input)  # read step output
        assert output[1].size() == (self.batch_size, self.output_dim)
