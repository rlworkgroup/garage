import tensorflow as tf
from garage.tf.core.networks import mlp
import numpy as np
from tests.fixtures import TfGraphTestCase


class TestNetworks(TfGraphTestCase):
    def setUp(self):
        super(TestNetworks, self).setUp()
        self.obs_input = np.array([[1, 2, 3, 4]])
        input_shape = self.obs_input.shape[1]  # 4

        self._input = tf.placeholder(
            tf.float32, shape=(None, input_shape), name="input")

        self._output_shape = 2

        self.mlp_f, self.mlp_all_layers_f = mlp(
            input_var=self._input,
            output_dim=self._output_shape,
            hidden_sizes=(32, 32),
            hidden_b_init=tf.constant_initializer(value=np.random.rand(1)),
            name="mlp",
            debug=True)

        self.sess.run(tf.global_variables_initializer())

    def test_input_shape(self):
        mlp_outputs = self.sess.run(
            [self.mlp_all_layers_f], feed_dict={self._input: self.obs_input})
        # first sample, first layer
        assert self._input.get_shape().as_list()[1] == mlp_outputs[0][0].shape[
            1]

    def test_output_shape(self):
        mlp_output = self.sess.run(
            [self.mlp_f], feed_dict={self._input: self.obs_input})
        # first sample
        assert mlp_output[0].shape[1] == self._output_shape

    def test_output_value(self):
        with tf.variable_scope("mlp", reuse=True):
            w = tf.get_variable("hidden_0/kernel")
            h1_w = self.sess.run(w)
            b = tf.get_variable("hidden_0/bias")
            h1_b = self.sess.run(b)
            w = tf.get_variable("hidden_1/kernel")
            h2_w = self.sess.run(w)
            b = tf.get_variable("hidden_1/bias")
            h2_b = self.sess.run(b)
            w = tf.get_variable("output/kernel")
            out_w = self.sess.run(w)
            b = tf.get_variable("output/bias")
            out_b = self.sess.run(b)

        mlp_output, mlp_outputs = self.sess.run(
            [self.mlp_f, self.mlp_all_layers_f],
            feed_dict={self._input: self.obs_input})

        h1_out = mlp_outputs[1]
        h2_out = mlp_outputs[2]
        out = mlp_outputs[3]

        # check first layer
        h2_in = np.dot(self.obs_input, h1_w) + h1_b
        h2_in = self.sess.run(tf.nn.relu(h2_in))
        np.testing.assert_array_almost_equal(h2_in, h1_out)
        # second layer
        h3_in = np.dot(h2_in, h2_w) + h2_b
        h3_in = self.sess.run(tf.nn.relu(h3_in))
        np.testing.assert_array_almost_equal(h3_in, h2_out)
        # output layer
        h3_out = np.dot(h3_in, out_w) + out_b
        np.testing.assert_array_almost_equal(h3_out, out)
        np.testing.assert_array_almost_equal(h3_out, mlp_output)
