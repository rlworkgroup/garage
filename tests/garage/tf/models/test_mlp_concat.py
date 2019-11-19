import numpy as np
import pytest
import tensorflow as tf

from garage.tf.models.mlp import mlp
from tests.fixtures import TfGraphTestCase


class TestMLPConcat(TfGraphTestCase):

    def setup_method(self):
        super(TestMLPConcat, self).setup_method()
        self.obs_input = np.array([[1, 2, 3, 4]])
        self.act_input = np.array([[1, 2, 3, 4]])
        input_shape_1 = self.obs_input.shape[1:]  # 4
        input_shape_2 = self.act_input.shape[1:]  # 4
        self.hidden_nonlinearity = tf.nn.relu

        self._obs_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, ) +
                                                   input_shape_1,
                                                   name='input')

        self._act_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, ) +
                                                   input_shape_2,
                                                   name='input')

        self._output_shape = 2

        # We build a default mlp
        with tf.compat.v1.variable_scope('MLP_Concat'):
            self.mlp_f = mlp(input_var=self._obs_input,
                             output_dim=self._output_shape,
                             hidden_sizes=(32, 32),
                             input_var2=self._act_input,
                             concat_layer=0,
                             hidden_nonlinearity=self.hidden_nonlinearity,
                             name='mlp1')

        self.sess.run(tf.compat.v1.global_variables_initializer())

    def test_multiple_same_mlp(self):
        # We create another mlp with the same name, trying to reuse it
        with tf.compat.v1.variable_scope('MLP_Concat', reuse=True):
            self.mlp_same_copy = mlp(
                input_var=self._obs_input,
                output_dim=self._output_shape,
                hidden_sizes=(32, 32),
                input_var2=self._act_input,
                concat_layer=0,
                hidden_nonlinearity=self.hidden_nonlinearity,
                name='mlp1')

        # We modify the weight of the default mlp and feed
        # The another mlp created should output the same result
        with tf.compat.v1.variable_scope('MLP_Concat', reuse=True):
            w = tf.compat.v1.get_variable('mlp1/hidden_0/kernel')
            self.sess.run(w.assign(w + 1))
            mlp_output = self.sess.run(self.mlp_f,
                                       feed_dict={
                                           self._obs_input: self.obs_input,
                                           self._act_input: self.act_input
                                       })
            mlp_output2 = self.sess.run(self.mlp_same_copy,
                                        feed_dict={
                                            self._obs_input: self.obs_input,
                                            self._act_input: self.act_input
                                        })

        np.testing.assert_array_almost_equal(mlp_output, mlp_output2)

    def test_different_mlp(self):
        # We create another mlp with different name
        with tf.compat.v1.variable_scope('MLP_Concat'):
            self.mlp_different_copy = mlp(
                input_var=self._obs_input,
                output_dim=self._output_shape,
                hidden_sizes=(32, 32),
                input_var2=self._act_input,
                concat_layer=0,
                hidden_nonlinearity=self.hidden_nonlinearity,
                name='mlp2')

        # Initialize the new mlp variables
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # We modify the weight of the default mlp and feed
        # The another mlp created should output different result
        with tf.compat.v1.variable_scope('MLP_Concat', reuse=True):
            w = tf.compat.v1.get_variable('mlp1/hidden_0/kernel')
            self.sess.run(w.assign(w + 1))
            mlp_output = self.sess.run(self.mlp_f,
                                       feed_dict={
                                           self._obs_input: self.obs_input,
                                           self._act_input: self.act_input
                                       })
            mlp_output2 = self.sess.run(self.mlp_different_copy,
                                        feed_dict={
                                            self._obs_input: self.obs_input,
                                            self._act_input: self.act_input
                                        })

        assert not np.array_equal(mlp_output, mlp_output2)

    def test_output_shape(self):
        mlp_output = self.sess.run(self.mlp_f,
                                   feed_dict={
                                       self._obs_input: self.obs_input,
                                       self._act_input: self.act_input
                                   })

        assert mlp_output.shape[1] == self._output_shape

    def test_output_value(self):
        with tf.compat.v1.variable_scope('MLP_Concat', reuse=True):
            h1_w = tf.compat.v1.get_variable('mlp1/hidden_0/kernel')
            h1_b = tf.compat.v1.get_variable('mlp1/hidden_0/bias')
            h2_w = tf.compat.v1.get_variable('mlp1/hidden_1/kernel')
            h2_b = tf.compat.v1.get_variable('mlp1/hidden_1/bias')
            out_w = tf.compat.v1.get_variable('mlp1/output/kernel')
            out_b = tf.compat.v1.get_variable('mlp1/output/bias')

        mlp_output = self.sess.run(self.mlp_f,
                                   feed_dict={
                                       self._obs_input: self.obs_input,
                                       self._act_input: self.act_input
                                   })

        # First layer
        h2_in = tf.matmul(tf.concat([self._obs_input, self._act_input], 1),
                          h1_w) + h1_b
        h2_in = self.hidden_nonlinearity(h2_in)

        # Second layer
        h3_in = tf.matmul(h2_in, h2_w) + h2_b
        h3_in = self.hidden_nonlinearity(h3_in)

        # Output layer
        h3_out = tf.matmul(h3_in, out_w) + out_b
        out = self.sess.run(h3_out,
                            feed_dict={
                                self._obs_input: self.obs_input,
                                self._act_input: self.act_input
                            })

        np.testing.assert_array_equal(out, mlp_output)

    def test_layer_normalization(self):
        # Create a mlp with layer normalization
        with tf.compat.v1.variable_scope('MLP_Concat'):
            self.mlp_f_w_n = mlp(input_var=self._obs_input,
                                 output_dim=self._output_shape,
                                 hidden_sizes=(32, 32),
                                 input_var2=self._act_input,
                                 concat_layer=0,
                                 hidden_nonlinearity=self.hidden_nonlinearity,
                                 name='mlp2',
                                 layer_normalization=True)

        # Initialize the new mlp variables
        self.sess.run(tf.compat.v1.global_variables_initializer())

        with tf.compat.v1.variable_scope('MLP_Concat', reuse=True):
            h1_w = tf.compat.v1.get_variable('mlp2/hidden_0/kernel')
            h1_b = tf.compat.v1.get_variable('mlp2/hidden_0/bias')
            h2_w = tf.compat.v1.get_variable('mlp2/hidden_1/kernel')
            h2_b = tf.compat.v1.get_variable('mlp2/hidden_1/bias')
            out_w = tf.compat.v1.get_variable('mlp2/output/kernel')
            out_b = tf.compat.v1.get_variable('mlp2/output/bias')
            beta_1 = tf.compat.v1.get_variable('mlp2/LayerNorm/beta')
            gamma_1 = tf.compat.v1.get_variable('mlp2/LayerNorm/gamma')
            beta_2 = tf.compat.v1.get_variable('mlp2/LayerNorm_1/beta')
            gamma_2 = tf.compat.v1.get_variable('mlp2/LayerNorm_1/gamma')

        # First layer
        y = tf.matmul(tf.concat([self._obs_input, self._act_input], 1),
                      h1_w) + h1_b
        y = self.hidden_nonlinearity(y)
        mean, variance = tf.nn.moments(y, [1], keep_dims=True)
        normalized_y = (y - mean) / tf.sqrt(variance + 1e-12)
        y_out = normalized_y * gamma_1 + beta_1

        # Second layer
        y = tf.matmul(y_out, h2_w) + h2_b
        y = self.hidden_nonlinearity(y)
        mean, variance = tf.nn.moments(y, [1], keep_dims=True)
        normalized_y = (y - mean) / tf.sqrt(variance + 1e-12)
        y_out = normalized_y * gamma_2 + beta_2

        # Output layer
        y = tf.matmul(y_out, out_w) + out_b

        out = self.sess.run(y,
                            feed_dict={
                                self._obs_input: self.obs_input,
                                self._act_input: self.act_input
                            })
        mlp_output = self.sess.run(self.mlp_f_w_n,
                                   feed_dict={
                                       self._obs_input: self.obs_input,
                                       self._act_input: self.act_input
                                   })

        np.testing.assert_array_almost_equal(out, mlp_output)

    @pytest.mark.parametrize('concat_idx', [2, 1, 0, -1, -2])
    def test_concat_layer(self, concat_idx):
        with tf.compat.v1.variable_scope('mlp_concat_test'):
            _ = mlp(input_var=self._obs_input,
                    output_dim=self._output_shape,
                    hidden_sizes=(64, 32),
                    input_var2=self._act_input,
                    concat_layer=concat_idx,
                    hidden_nonlinearity=self.hidden_nonlinearity,
                    name='mlp2')
        obs_input_size = self._obs_input.shape[1].value
        act_input_size = self._act_input.shape[1].value

        expected_units = [obs_input_size, 64, 32]
        expected_units[concat_idx] += act_input_size

        actual_units = []
        with tf.compat.v1.variable_scope('mlp_concat_test', reuse=True):
            h1_w = tf.compat.v1.get_variable('mlp2/hidden_0/kernel')
            h2_w = tf.compat.v1.get_variable('mlp2/hidden_1/kernel')
            out_w = tf.compat.v1.get_variable('mlp2/output/kernel')

            actual_units.append(h1_w.shape[0].value)
            actual_units.append(h2_w.shape[0].value)
            actual_units.append(out_w.shape[0].value)

        assert np.array_equal(expected_units, actual_units)

    @pytest.mark.parametrize('concat_idx', [2, 1, 0, -1, -2])
    def test_invalid_concat_args(self, concat_idx):
        with tf.compat.v1.variable_scope('mlp_concat_test'):
            _ = mlp(input_var=self._obs_input,
                    output_dim=self._output_shape,
                    hidden_sizes=(64, 32),
                    concat_layer=concat_idx,
                    hidden_nonlinearity=self.hidden_nonlinearity,
                    name='mlp_no_input2')

        obs_input_size = self._obs_input.shape[1].value

        # concat_layer argument should be silently ignored.
        expected_units = [obs_input_size, 64, 32]

        actual_units = []
        with tf.compat.v1.variable_scope('mlp_concat_test', reuse=True):
            h1_w = tf.compat.v1.get_variable('mlp_no_input2/hidden_0/kernel')
            h2_w = tf.compat.v1.get_variable('mlp_no_input2/hidden_1/kernel')
            out_w = tf.compat.v1.get_variable('mlp_no_input2/output/kernel')

            actual_units.append(h1_w.shape[0].value)
            actual_units.append(h2_w.shape[0].value)
            actual_units.append(out_w.shape[0].value)

        assert np.array_equal(expected_units, actual_units)

    @pytest.mark.parametrize('concat_idx', [2, 1, 0, -1, -2])
    def test_no_hidden(self, concat_idx):
        with tf.compat.v1.variable_scope('mlp_concat_test'):
            _ = mlp(input_var=self._obs_input,
                    output_dim=self._output_shape,
                    hidden_sizes=(),
                    input_var2=self._act_input,
                    concat_layer=concat_idx,
                    hidden_nonlinearity=self.hidden_nonlinearity,
                    name='mlp2')

        obs_input_size = self._obs_input.shape[1].value
        act_input_size = self._act_input.shape[1].value

        # concat_layer argument should be reset to point to input_var.
        expected_units = [obs_input_size]
        expected_units[0] += act_input_size

        actual_units = []
        with tf.compat.v1.variable_scope('mlp_concat_test', reuse=True):
            out_w = tf.compat.v1.get_variable('mlp2/output/kernel')
            actual_units.append(out_w.shape[0].value)

        assert np.array_equal(expected_units, actual_units)
