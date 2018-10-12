import unittest

import tensorflow as tf
from garage.tf.core.networks import MLPs
import numpy as np


class TestNetworks(unittest.TestCase):
    def test_networks(self):
        obs_input = np.array([[1, 2, 3, 4]])
        obs_dim = obs_input.shape[1]
        action_dim = 2
        activation_f = tf.nn.relu
        hidden_sizes = (32, 32)
        bias = np.random.rand(1)

        with tf.variable_scope("Test"):
            obs_t_ph = tf.placeholder(
                tf.float32, [None, obs_dim], name="input")

            mlps = MLPs(
                name="mlps",
                input_shape=(obs_dim, ),
                output_dim=action_dim,
                input_var=obs_t_ph,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=activation_f,
                output_nonlinearity=None,
                hidden_b_init=tf.constant_initializer(value=bias))

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                with tf.variable_scope("mlps", reuse=True):
                    w = tf.get_variable("hidden_0/kernel")
                    h1_w = sess.run(w)
                    b = tf.get_variable("hidden_0/bias")
                    h1_b = sess.run(b)
                    w = tf.get_variable("hidden_1/kernel")
                    h2_w = sess.run(w)
                    b = tf.get_variable("hidden_1/bias")
                    h2_b = sess.run(b)
                    w = tf.get_variable("output/kernel")
                    out_w = sess.run(w)
                    b = tf.get_variable("output/bias")
                    out_b = sess.run(b)

                # get output from session.run
                layer_outputs = sess.run(
                    mlps.layers, feed_dict={obs_t_ph: obs_input})

                h1_out = layer_outputs[1]
                h2_out = layer_outputs[2]
                out = layer_outputs[3]

                # check first layer
                h2_in = np.dot(obs_input, h1_w) + h1_b
                h2_in = sess.run(activation_f(h2_in))
                np.testing.assert_array_almost_equal(h2_in, h1_out)

                # second layer
                h3_in = np.dot(h2_in, h2_w) + h2_b
                h3_in = sess.run(activation_f(h3_in))
                np.testing.assert_array_almost_equal(h3_in, h2_out)

                # output layer
                h3_out = np.dot(h3_in, out_w) + out_b
                np.testing.assert_array_almost_equal(h3_out, out)
