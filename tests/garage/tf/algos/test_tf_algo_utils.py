"""
Test tf algo utility functions mainly in garage.tf._utils
"""
import numpy as np
import tensorflow as tf

import garage.tf.algos._utils as tf_algo_utils
from tests.fixtures import TfGraphTestCase


class TestAlgoUtil(TfGraphTestCase):
    """Test class for tf algo utility functions."""

    def test_make_optimizer_with_type(self):
        """Test make_optimizer function with type as first argument."""
        optimizer_type = tf.compat.v1.train.AdamOptimizer
        lr = 0.123
        optimizer = tf_algo_utils.make_optimizer(optimizer_type,
                                                 learning_rate=lr,
                                                 name='testOptimizer')
        assert isinstance(optimizer, optimizer_type)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        assert optimizer._name == 'testOptimizer'
        assert np.allclose(
            optimizer._lr, lr
        )  # Adam holds the value of learning rate in private variable self._lr
