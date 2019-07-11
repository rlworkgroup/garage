import numpy as np
import tensorflow as tf

from garage.tf.core.parameter import parameter
from tests.fixtures import TfGraphTestCase


class TestParameter(TfGraphTestCase):
    def test_param(self):
        input_vars = tf.placeholder(shape=[None, 2, 3, 4], dtype=tf.float32)
        initial_params = np.array([48, 21, 33])

        reshaped_param, param = parameter(
            input_var=input_vars,
            length=3,
            initializer=tf.constant_initializer(initial_params))

        data = np.zeros(shape=[5, 2, 3, 4])
        feed_dict = {
            input_vars: data,
        }

        self.sess.run(tf.global_variables_initializer())
        p = self.sess.run(reshaped_param, feed_dict=feed_dict)

        assert p.shape[:-1] == data.shape[:-1]
        assert np.all(p[0, 0, 0, :] == initial_params)

        p = self.sess.run(param, feed_dict=feed_dict)

        assert p.shape == (3, )
        assert np.all(p == initial_params)
