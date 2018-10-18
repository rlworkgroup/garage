import unittest

import lasagne.layers as L
import numpy as np

from garage.theano.core.network import GRUNetwork
from garage.theano.misc import tensor_utils


class TestNetworks(unittest.TestCase):
    def test_gru_network(self):
        network = GRUNetwork(
            input_shape=(2, 3),
            output_dim=5,
            hidden_dim=4,
        )
        f_output = tensor_utils.compile_function(
            inputs=[network.input_layer.input_var],
            outputs=L.get_output(network.output_layer))
        assert f_output(np.zeros((6, 8, 2, 3))).shape == (6, 8, 5)
