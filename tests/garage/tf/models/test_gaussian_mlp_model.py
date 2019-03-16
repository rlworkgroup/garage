import numpy as np
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.distributions import DiagonalGaussian
from garage.tf.models.gaussian_mlp_model import GaussianMLPModel
from tests.fixtures import TfGraphTestCase


class TestGaussianMLPModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.input_var = tf.placeholder(tf.float32, shape=(None, 5))
        self.obs = np.ones((1, 5))
        self.output_dim = 2

    def test_std_share_network(self):
        model = GaussianMLPModel(
            output_dim=self.output_dim, std_share_network=True)
        outputs = model.build(self.input_var)
        out, model_out = self.sess.run(
            [outputs[:-1], model.networks['default'].outputs[:-1]],
            feed_dict={model.networks['default'].input: self.obs})

        assert np.array_equal(out, model_out)

        # --- output shape ---
        # should be 2 * output_dim
        with tf.variable_scope(model.name, reuse=True):
            std_share_output_weights = tf.get_variable(
                'dist_params/mean_std_network/output/kernel')
            std_share_output_bias = tf.get_variable(
                'dist_params/mean_std_network/output/bias')
        assert std_share_output_weights.shape[1] == self.output_dim * 2
        assert std_share_output_bias.shape == self.output_dim * 2

        action, mean, log_std = out
        # --- action ---
        rnd = tf.random.normal(
            shape=mean.shape[1:], seed=deterministic.get_seed()).eval()
        expected_action = rnd * np.exp(log_std) + mean

        assert np.array_equal(action, expected_action)

    def test_without_std_share_network(self):
        model = GaussianMLPModel(
            output_dim=self.output_dim, std_share_network=False, init_std=1.1)
        outputs = model.build(self.input_var)
        out, model_out = self.sess.run(
            [outputs[:-1], model.networks['default'].outputs[:-1]],
            feed_dict={model.networks['default'].input: self.obs})

        assert np.array_equal(out, model_out)
        # --- output shape ---
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

        action, mean, log_std = out
        # --- log_std ---
        assert np.allclose(log_std, np.log(1.1))

        # --- action ---
        rnd = tf.random.normal(
            shape=mean.shape[1:], seed=deterministic.get_seed()).eval()
        expected_action = rnd * np.exp(log_std) + mean

        assert np.array_equal(action, expected_action)

    def test_adaptive_std(self):
        model = GaussianMLPModel(output_dim=self.output_dim, adaptive_std=True)
        outputs = model.build(self.input_var)
        out, model_out = self.sess.run(
            [outputs[:-1], model.networks['default'].outputs[:-1]],
            feed_dict={model.networks['default'].input: self.obs})

        assert np.array_equal(out, model_out)
        # --- output shape ---
        # output shape should be output_dim
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

        action, mean, log_std = out
        # --- action ---
        rnd = tf.random.normal(
            shape=mean.shape[1:], seed=deterministic.get_seed()).eval()
        expected_action = rnd * np.exp(log_std) + mean

        assert np.array_equal(action, expected_action)

    def test_dist(self):
        model = GaussianMLPModel(
            output_dim=self.output_dim, std_share_network=True)
        _, _, _, dist = model.build(self.input_var)
        assert isinstance(dist, DiagonalGaussian)
        assert dist.dim == self.output_dim
