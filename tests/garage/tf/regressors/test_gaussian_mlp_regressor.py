import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.optimizers import PenaltyLbfgsOptimizer
from garage.tf.regressors import GaussianMLPRegressor
from tests.fixtures import TfGraphTestCase


class TestGaussianMLPRegressor(TfGraphTestCase):
    # unmarked to balance test jobs
    # @pytest.mark.large
    def test_fit_normalized(self):
        gmr = GaussianMLPRegressor(input_shape=(1, ), output_dim=1)
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{'observations': [[x]], 'returns': [np.sin(x)]} for x in data]

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        returns = returns.reshape((-1, 1))
        for _ in range(150):
            gmr.fit(observations, returns)

        paths = {
            'observations': [[-np.pi], [-np.pi / 2], [-np.pi / 4], [0],
                             [np.pi / 4], [np.pi / 2], [np.pi]]
        }

        prediction = gmr.predict(paths['observations'])

        expected = [[0], [-1], [-0.707], [0], [0.707], [1], [0]]
        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(gmr.model.networks['default'].x_mean)
        x_mean_expected = np.mean(observations, axis=0, keepdims=True)
        x_std = self.sess.run(gmr.model.networks['default'].x_std)
        x_std_expected = np.std(observations, axis=0, keepdims=True)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

        y_mean = self.sess.run(gmr.model.networks['default'].y_mean)
        y_mean_expected = np.mean(returns, axis=0, keepdims=True)
        y_std = self.sess.run(gmr.model.networks['default'].y_std)
        y_std_expected = np.std(returns, axis=0, keepdims=True)

        assert np.allclose(y_mean, y_mean_expected)
        assert np.allclose(y_std, y_std_expected)

    # unmarked to balance test jobs
    # @pytest.mark.large
    def test_fit_unnormalized(self):
        gmr = GaussianMLPRegressor(input_shape=(1, ),
                                   output_dim=1,
                                   subsample_factor=0.9,
                                   normalize_inputs=False,
                                   normalize_outputs=False)
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{'observations': [[x]], 'returns': [np.sin(x)]} for x in data]

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        for _ in range(150):
            gmr.fit(observations, returns.reshape((-1, 1)))

        paths = {
            'observations': [[-np.pi], [-np.pi / 2], [-np.pi / 4], [0],
                             [np.pi / 4], [np.pi / 2], [np.pi]]
        }

        prediction = gmr.predict(paths['observations'])

        expected = [[0], [-1], [-0.707], [0], [0.707], [1], [0]]
        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(gmr.model.networks['default'].x_mean)
        x_mean_expected = np.zeros_like(x_mean)
        x_std = self.sess.run(gmr.model.networks['default'].x_std)
        x_std_expected = np.ones_like(x_std)
        assert np.array_equal(x_mean, x_mean_expected)
        assert np.array_equal(x_std, x_std_expected)

        y_mean = self.sess.run(gmr.model.networks['default'].y_mean)
        y_mean_expected = np.zeros_like(y_mean)
        y_std = self.sess.run(gmr.model.networks['default'].y_std)
        y_std_expected = np.ones_like(y_std)

        assert np.allclose(y_mean, y_mean_expected)
        assert np.allclose(y_std, y_std_expected)

    # unmarked to balance test jobs
    # @pytest.mark.large
    def test_fit_smaller_subsample_factor(self):
        gmr = GaussianMLPRegressor(input_shape=(1, ),
                                   output_dim=1,
                                   subsample_factor=0.9)
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{'observations': [[x]], 'returns': [np.sin(x)]} for x in data]

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        for _ in range(150):
            gmr.fit(observations, returns.reshape((-1, 1)))

        paths = {
            'observations': [[-np.pi], [-np.pi / 2], [-np.pi / 4], [0],
                             [np.pi / 4], [np.pi / 2], [np.pi]]
        }

        prediction = gmr.predict(paths['observations'])

        expected = [[0], [-1], [-0.707], [0], [0.707], [1], [0]]
        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

    # unmarked to balance test jobs
    # @pytest.mark.large
    def test_fit_without_trusted_region(self):
        gmr = GaussianMLPRegressor(input_shape=(1, ),
                                   output_dim=1,
                                   use_trust_region=False)
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{'observations': [[x]], 'returns': [np.sin(x)]} for x in data]

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        for _ in range(150):
            gmr.fit(observations, returns.reshape((-1, 1)))

        paths = {
            'observations': [[-np.pi], [-np.pi / 2], [-np.pi / 4], [0],
                             [np.pi / 4], [np.pi / 2], [np.pi]]
        }

        prediction = gmr.predict(paths['observations'])

        expected = [[0], [-1], [-0.707], [0], [0.707], [1], [0]]
        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

    @pytest.mark.parametrize('output_dim, input_shape',
                             [(1, (1, )), (1, (2, )), (2, (3, )), (2, (1, 1)),
                              (3, (2, 2))])
    def test_log_likelihood_sym(self, output_dim, input_shape):
        gmr = GaussianMLPRegressor(input_shape=input_shape,
                                   output_dim=output_dim,
                                   optimizer=PenaltyLbfgsOptimizer,
                                   optimizer_args=dict())

        new_input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, ) + input_shape)
        new_ys_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                              name='ys',
                                              shape=(None, output_dim))

        data = np.random.random(size=input_shape)
        label = np.ones(output_dim)

        outputs = gmr.log_likelihood_sym(new_input_var,
                                         new_ys_var,
                                         name='ll_sym')
        ll_from_sym = self.sess.run(outputs,
                                    feed_dict={
                                        new_input_var: [data],
                                        new_ys_var: [label]
                                    })
        mean, log_std = gmr._f_pdists([data])
        ll = gmr.model.networks['default'].dist.log_likelihood(
            [label], dict(mean=mean, log_std=log_std))
        assert np.allclose(ll, ll_from_sym, rtol=0, atol=1e-5)

    def test_is_pickleable(self):
        gmr = GaussianMLPRegressor(input_shape=(1, ), output_dim=1)

        with tf.compat.v1.variable_scope(
                'GaussianMLPRegressor/GaussianMLPRegressorModel', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/hidden_0/bias')
        bias.load(tf.ones_like(bias).eval())

        result1 = gmr.predict(np.ones((1, 1)))
        h = pickle.dumps(gmr)

        with tf.compat.v1.Session(graph=tf.Graph()):
            gmr_pickled = pickle.loads(h)
            result2 = gmr_pickled.predict(np.ones((1, 1)))
            assert np.array_equal(result1, result2)

    def test_is_pickleable2(self):
        gmr = GaussianMLPRegressor(input_shape=(1, ), output_dim=1)

        with tf.compat.v1.variable_scope(
                'GaussianMLPRegressor/GaussianMLPRegressorModel', reuse=True):
            x_mean = tf.compat.v1.get_variable('normalized_vars/x_mean')
        x_mean.load(tf.ones_like(x_mean).eval())
        x1 = gmr.model.networks['default'].x_mean.eval()
        h = pickle.dumps(gmr)
        with tf.compat.v1.Session(graph=tf.Graph()):
            gmr_pickled = pickle.loads(h)
            x2 = gmr_pickled.model.networks['default'].x_mean.eval()
            assert np.array_equal(x1, x2)
