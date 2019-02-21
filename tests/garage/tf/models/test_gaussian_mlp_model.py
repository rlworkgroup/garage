import numpy as np
import tensorflow as tf

from garage.tf.models.gaussian_mlp_model import GaussianMLPModel
from garage.tf.models.gaussian_mlp_model2 import GaussianMLPModel2
from tests.fixtures import TfGraphTestCase


class TestGaussianMLPModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.input_var = tf.placeholder(tf.float32, shape=(None, 5))
        self.obs = np.ones((1, 5))

    def test_gaussian_mlp_model_with_std_share_network(self):
        model = GaussianMLPModel(output_dim=2, std_share_network=True)
        outputs = model.build(self.input_var)
        out, model_out = self.sess.run(
            [outputs[:-1], model.networks['default'].outputs[:-1]],
            feed_dict={model.networks['default'].input: self.obs})

        assert np.array_equal(out, model_out)
        # output shape should be 2 * output_dim
        with tf.variable_scope(model.name, reuse=True):
            std_share_output_weights = tf.get_variable(
                'dist_params/mean_std_network/output/kernel')
        assert std_share_output_weights.shape[1] == 4

    def test_gaussian_mlp_model_without_std_share_network(self):
        model = GaussianMLPModel(output_dim=2, std_share_network=False)
        outputs = model.build(self.input_var)
        out, model_out = self.sess.run(
            [outputs[:-1], model.networks['default'].outputs[:-1]],
            feed_dict={model.networks['default'].input: self.obs})

        assert np.array_equal(out, model_out)
        # output shape should be output_dim
        with tf.variable_scope(model.name, reuse=True):
            mean_output_weights = tf.get_variable(
                'dist_params/mean_network/output/kernel')
            std_output_weights = tf.get_variable(
                'dist_params/std_network/parameter')
        assert mean_output_weights.shape[1] == 2
        assert std_output_weights.shape == 2

    def test_gaussian_mlp_model_with_adaptive_std(self):
        model = GaussianMLPModel(output_dim=2, adaptive_std=True)
        outputs = model.build(self.input_var)
        out, model_out = self.sess.run(
            [outputs[:-1], model.networks['default'].outputs[:-1]],
            feed_dict={model.networks['default'].input: self.obs})

        assert np.array_equal(out, model_out)
        # output shape should be output_dim
        with tf.variable_scope(model.name, reuse=True):
            mean_output_weights = tf.get_variable(
                'dist_params/mean_network/output/kernel')
            std_output_weights = tf.get_variable(
                'dist_params/std_network/output/kernel')
        assert mean_output_weights.shape[1] == 2
        assert std_output_weights.shape[1] == 2

    ############
    def test_gaussian_mlp_model2_with_std_share_network(self):
        model = GaussianMLPModel2(output_dim=2, std_share_network=True)
        outputs = model.build(self.input_var)
        out, model_out = self.sess.run(
            [outputs[:-1], model.networks['default'].outputs[:-1]],
            feed_dict={model.networks['default'].input: self.obs})

        assert np.array_equal(out, model_out)
        # output shape should be 2 * output_dim
        with tf.variable_scope(model.name, reuse=True):
            std_share_output_weights = tf.get_variable(
                'dist_params/mean_std_network/output/kernel')
        assert std_share_output_weights.shape[1] == 4

    def test_gaussian_mlp_model2_without_std_share_network(self):
        model = GaussianMLPModel2(output_dim=2, std_share_network=False)
        outputs = model.build(self.input_var)
        out, model_out = self.sess.run(
            [outputs[:-1], model.networks['default'].outputs[:-1]],
            feed_dict={model.networks['default'].input: self.obs})

        assert np.array_equal(out, model_out)
        # output shape should be output_dim
        with tf.variable_scope(model.name, reuse=True):
            mean_output_weights = tf.get_variable(
                'dist_params/mean_network/output/kernel')
            std_output_weights = tf.get_variable(
                'dist_params/std_network/parameter')
        assert mean_output_weights.shape[1] == 2
        assert std_output_weights.shape == 2

    def test_gaussian_mlp_model2_with_adaptive_std(self):
        model = GaussianMLPModel2(output_dim=2, adaptive_std=True)
        outputs = model.build(self.input_var)
        out, model_out = self.sess.run(
            [outputs[:-1], model.networks['default'].outputs[:-1]],
            feed_dict={model.networks['default'].input: self.obs})

        assert np.array_equal(out, model_out)
        # output shape should be output_dim
        with tf.variable_scope(model.name, reuse=True):
            mean_output_weights = tf.get_variable(
                'dist_params/mean_network/output/kernel')
            std_output_weights = tf.get_variable(
                'dist_params/std_network/output/kernel')
        assert mean_output_weights.shape[1] == 2
        assert std_output_weights.shape[1] == 2
