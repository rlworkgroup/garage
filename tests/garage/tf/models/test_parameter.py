import numpy as np
import tensorflow as tf

from garage.tf.models.parameter import parameter
from garage.tf.models.parameter import recurrent_parameter
from tests.fixtures import TfGraphTestCase


class TestParameter(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.input_vars = tf.placeholder(shape=[None, 2, 5], dtype=tf.float32)
        self.step_input_vars = tf.placeholder(shape=[None, 5],
                                              dtype=tf.float32)
        self.initial_params = np.array([48, 21, 33])

        self.data = np.zeros(shape=[5, 2, 5])
        self.step_data = np.zeros(shape=[5, 5])
        self.feed_dict = {
            self.input_vars: self.data,
            self.step_input_vars: self.step_data
        }

    def test_param(self):
        param = parameter(input_var=self.input_vars,
                          length=3,
                          initializer=tf.constant_initializer(
                              self.initial_params))
        self.sess.run(tf.global_variables_initializer())
        p = self.sess.run(param, feed_dict=self.feed_dict)

        assert p.shape == (5, 3)
        assert np.all(p == self.initial_params)

    def test_recurrent_param(self):
        param, _ = recurrent_parameter(input_var=self.input_vars,
                                       step_input_var=self.step_input_vars,
                                       length=3,
                                       initializer=tf.constant_initializer(
                                           self.initial_params))
        self.sess.run(tf.compat.v1.global_variables_initializer())
        p = self.sess.run(param, feed_dict=self.feed_dict)

        assert p.shape == (5, 2, 3)
        assert np.array_equal(p, np.full([5, 2, 3], self.initial_params))
