import numpy as np
import tensorflow as tf

from garage.tf.models.mlp import mlp

from tests.fixtures import TfGraphTestCase


class TestMLP(TfGraphTestCase):

    # pylint: disable=unsubscriptable-object
    def setup_method(self):
        super(TestMLP, self).setup_method()
        self.obs_input = np.array([[1, 2, 3, 4]])
        input_shape = self.obs_input.shape[1:]  # 4
        self.hidden_nonlinearity = tf.nn.relu

        self._input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, ) + input_shape,
                                               name='input')

        self._output_shape = 2

        # We build a default mlp
        with tf.compat.v1.variable_scope('MLP'):
            self.mlp_f = mlp(input_var=self._input,
                             output_dim=self._output_shape,
                             hidden_sizes=(32, 32),
                             hidden_nonlinearity=self.hidden_nonlinearity,
                             name='mlp1')

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def test_multiple_same_mlp(self):
        # We create another mlp with the same name, trying to reuse it
        with tf.compat.v1.variable_scope('MLP', reuse=True):
            self.mlp_same_copy = mlp(
                input_var=self._input,
                output_dim=self._output_shape,
                hidden_sizes=(32, 32),
                hidden_nonlinearity=self.hidden_nonlinearity,
                name='mlp1')

        # We modify the weight of the default mlp and feed
        # The another mlp created should output the same result
        with tf.compat.v1.variable_scope('MLP', reuse=True):
            w = tf.compat.v1.get_variable('mlp1/hidden_0/kernel')
            self.sess.run(w.assign(w + 1))
            mlp_output = self.sess.run(self.mlp_f,
                                       feed_dict={self._input: self.obs_input})
            mlp_output2 = self.sess.run(
                self.mlp_same_copy, feed_dict={self._input: self.obs_input})

        np.testing.assert_array_almost_equal(mlp_output, mlp_output2)

    def test_different_mlp(self):
        # We create another mlp with different name
        with tf.compat.v1.variable_scope('MLP'):
            self.mlp_different_copy = mlp(
                input_var=self._input,
                output_dim=self._output_shape,
                hidden_sizes=(32, 32),
                hidden_nonlinearity=self.hidden_nonlinearity,
                name='mlp2')

        # Initialize the new mlp variables
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # We modify the weight of the default mlp and feed
        # The another mlp created should output different result
        with tf.compat.v1.variable_scope('MLP', reuse=True):
            w = tf.compat.v1.get_variable('mlp1/hidden_0/kernel')
            self.sess.run(w.assign(w + 1))
            mlp_output = self.sess.run(self.mlp_f,
                                       feed_dict={self._input: self.obs_input})
            mlp_output2 = self.sess.run(
                self.mlp_different_copy,
                feed_dict={self._input: self.obs_input})

        np.not_equal(mlp_output, mlp_output2)

    def test_output_shape(self):
        mlp_output = self.sess.run(self.mlp_f,
                                   feed_dict={self._input: self.obs_input})

        assert mlp_output.shape[1] == self._output_shape

    def test_output_value(self):
        with tf.compat.v1.variable_scope('MLP', reuse=True):
            h1_w = tf.compat.v1.get_variable('mlp1/hidden_0/kernel')
            h1_b = tf.compat.v1.get_variable('mlp1/hidden_0/bias')
            h2_w = tf.compat.v1.get_variable('mlp1/hidden_1/kernel')
            h2_b = tf.compat.v1.get_variable('mlp1/hidden_1/bias')
            out_w = tf.compat.v1.get_variable('mlp1/output/kernel')
            out_b = tf.compat.v1.get_variable('mlp1/output/bias')

        mlp_output = self.sess.run(self.mlp_f,
                                   feed_dict={self._input: self.obs_input})

        # First layer
        h2_in = tf.matmul(self._input, h1_w) + h1_b
        h2_in = self.hidden_nonlinearity(h2_in)

        # Second layer
        h3_in = tf.matmul(h2_in, h2_w) + h2_b
        h3_in = self.hidden_nonlinearity(h3_in)

        # Output layer
        h3_out = tf.matmul(h3_in, out_w) + out_b
        out = self.sess.run(h3_out, feed_dict={self._input: self.obs_input})

        np.testing.assert_array_equal(out, mlp_output)

    def test_layer_normalization(self):
        # Create a mlp with layer normalization
        with tf.compat.v1.variable_scope('MLP'):
            self.mlp_f_w_n = mlp(input_var=self._input,
                                 output_dim=self._output_shape,
                                 hidden_sizes=(32, 32),
                                 hidden_nonlinearity=self.hidden_nonlinearity,
                                 name='mlp2',
                                 layer_normalization=True)

        # Initialize the new mlp variables
        self.sess.run(tf.compat.v1.global_variables_initializer())

        with tf.compat.v1.variable_scope('MLP', reuse=True):
            h1_w = tf.compat.v1.get_variable('mlp2/hidden_0/kernel')
            h1_b = tf.compat.v1.get_variable('mlp2/hidden_0/bias')
            h2_w = tf.compat.v1.get_variable('mlp2/hidden_1/kernel')
            h2_b = tf.compat.v1.get_variable('mlp2/hidden_1/bias')
            out_w = tf.compat.v1.get_variable('mlp2/output/kernel')
            out_b = tf.compat.v1.get_variable('mlp2/output/bias')

        with tf.compat.v1.variable_scope('MLP_1', reuse=True) as vs:
            gamma_1, beta_1, gamma_2, beta_2 = vs.global_variables()

        # First layer
        y = tf.matmul(self._input, h1_w) + h1_b
        y = self.hidden_nonlinearity(y)
        mean, variance = tf.nn.moments(y, [1], keepdims=True)
        normalized_y = (y - mean) / tf.sqrt(variance + 1e-12)
        y_out = normalized_y * gamma_1 + beta_1

        # Second layer
        y = tf.matmul(y_out, h2_w) + h2_b
        y = self.hidden_nonlinearity(y)
        mean, variance = tf.nn.moments(y, [1], keepdims=True)
        normalized_y = (y - mean) / tf.sqrt(variance + 1e-12)
        y_out = normalized_y * gamma_2 + beta_2

        # Output layer
        y = tf.matmul(y_out, out_w) + out_b

        out = self.sess.run(y, feed_dict={self._input: self.obs_input})
        mlp_output = self.sess.run(self.mlp_f_w_n,
                                   feed_dict={self._input: self.obs_input})

        np.testing.assert_array_almost_equal(out, mlp_output, decimal=2)
