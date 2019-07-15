import numpy as np
import tensorflow as tf

from garage.tf.core.parameter import parameter
from garage.tf.misc.tensor_utils import broadcast_with_batch
from tests.fixtures import TfGraphTestCase


class TestParameter(TfGraphTestCase):
    def setup_method(self):
        super().setup_method()
        self.input_vars = tf.placeholder(
            shape=[None, 2, 3, 4], dtype=tf.float32)
        self.initial_params = np.array([48, 21, 33])

        self.param = parameter(
            length=3, initializer=tf.constant_initializer(self.initial_params))

        self.data = np.zeros(shape=[5, 2, 3, 4])
        self.feed_dict = {
            self.input_vars: self.data,
        }
        self.sess.run(tf.global_variables_initializer())

    def test_param(self):
        p = self.sess.run(self.param, feed_dict=self.feed_dict)

        assert p.shape == (3, )
        assert np.all(p == self.initial_params)

    def test_broadcast_with_batch(self):
        broadcast_param = broadcast_with_batch(
            self.param, self.input_vars, batch_dim=3)
        p = self.sess.run(broadcast_param, feed_dict=self.feed_dict)

        assert p.shape[:-1] == self.data.shape[:-1]
        assert np.all(p[0, 0, 0, :] == self.initial_params)
