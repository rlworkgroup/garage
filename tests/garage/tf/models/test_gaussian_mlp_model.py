from unittest import mock

import numpy as np
import tensorflow as tf

from garage.tf.models import GaussianMLPModel
from tests.fixtures import TfGraphTestCase


class TestGaussianMLPModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.input_var = tf.placeholder(tf.float32, shape=(None, 5))
        self.obs = np.ones((1, 5))
        self.output_dim = 1

    @mock.patch('tensorflow.random.normal')
    def test_std_share_network_output_values(self, mock_normal):
        mock_normal.return_value = 0.5
        model = GaussianMLPModel(
            output_dim=self.output_dim,
            hidden_sizes=(2, ),
            std_share_network=True,
            hidden_nonlinearity=None,
            hidden_w_init=tf.ones_initializer,
            output_w_init=tf.ones_initializer)
        outputs = model.build(self.input_var)
        action, mean, log_std = self.sess.run(
            outputs[:-1], feed_dict={self.input_var: self.obs})

        assert mean == log_std == 10.
        assert action == np.exp(10.) * 0.5 + 10

    def test_std_share_network_shapes(self):
        # should be 2 * output_dim
        model = GaussianMLPModel(
            output_dim=self.output_dim, std_share_network=True)
        model.build(self.input_var)
        with tf.variable_scope(model.name, reuse=True):
            std_share_output_weights = tf.get_variable(
                'dist_params/mean_std_network/output/kernel')
            std_share_output_bias = tf.get_variable(
                'dist_params/mean_std_network/output/bias')
        assert std_share_output_weights.shape[1] == self.output_dim * 2
        assert std_share_output_bias.shape == self.output_dim * 2

    @mock.patch('tensorflow.random.normal')
    def test_without_std_share_network_output_values(self, mock_normal):
        mock_normal.return_value = 0.5

        model = GaussianMLPModel(
            output_dim=self.output_dim,
            hidden_sizes=(2, ),
            init_std=2,
            std_share_network=False,
            adaptive_std=False,
            hidden_nonlinearity=None,
            hidden_w_init=tf.ones_initializer,
            output_w_init=tf.ones_initializer)
        outputs = model.build(self.input_var)
        action, mean, log_std = self.sess.run(
            outputs[:-1], feed_dict={self.input_var: self.obs})

        assert mean == 10.
        assert log_std == np.log(2.)
        assert action == np.exp(np.log(2.)) * 0.5 + 10

    def test_without_std_share_network_shapes(self):
        model = GaussianMLPModel(
            output_dim=self.output_dim,
            std_share_network=False,
            adaptive_std=False)
        model.build(self.input_var)
        # output shape should be output_dim
        with tf.variable_scope(model.name, reuse=True):
            mean_output_weights = tf.get_variable(
                'dist_params/mean_network/output/kernel')
            mean_output_bias = tf.get_variable(
                'dist_params/mean_network/output/bias')
            log_std_output_weights = tf.get_variable(
                'dist_params/log_std_network/parameter')
        assert mean_output_weights.shape[1] == self.output_dim
        assert mean_output_bias.shape == self.output_dim
        assert log_std_output_weights.shape == self.output_dim

    @mock.patch('tensorflow.random.normal')
    def test_adaptive_std_network_output_values(self, mock_normal):
        mock_normal.return_value = 0.5
        model = GaussianMLPModel(
            output_dim=self.output_dim,
            std_share_network=False,
            hidden_sizes=(2, ),
            std_hidden_sizes=(2, ),
            adaptive_std=True,
            hidden_nonlinearity=None,
            hidden_w_init=tf.ones_initializer,
            output_w_init=tf.ones_initializer,
            std_hidden_nonlinearity=None,
            std_hidden_w_init=tf.ones_initializer,
            std_output_w_init=tf.ones_initializer)
        model.build(self.input_var)
        action, mean, log_std = self.sess.run(
            model.networks['default'].outputs[:-1],
            feed_dict={model.networks['default'].input: self.obs})

        assert mean == 10.0
        assert log_std == 10.0
        assert action == np.exp(10) * 0.5 + 10

    def test_adaptive_std_output_shape(self):
        # output shape should be output_dim
        model = GaussianMLPModel(
            output_dim=self.output_dim,
            std_share_network=False,
            adaptive_std=True)
        model.build(self.input_var)
        with tf.variable_scope(model.name, reuse=True):
            mean_output_weights = tf.get_variable(
                'dist_params/mean_network/output/kernel')
            mean_output_bias = tf.get_variable(
                'dist_params/mean_network/output/bias')
            log_std_output_weights = tf.get_variable(
                'dist_params/log_std_network/output/kernel')
            log_std_output_bias = tf.get_variable(
                'dist_params/log_std_network/output/bias')

        assert mean_output_weights.shape[1] == self.output_dim
        assert mean_output_bias.shape == self.output_dim
        assert log_std_output_weights.shape[1] == self.output_dim
        assert log_std_output_bias.shape == self.output_dim
