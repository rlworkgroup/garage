from nose2.tools.params import params
import numpy as np
import tensorflow as tf

from garage.tf.optimizers import PenaltyLbfgsOptimizer
from garage.tf.regressors import GaussianMLPRegressorWithModel
from tests.fixtures import TfGraphTestCase


class TestGaussianMLPRegressorWithModel(TfGraphTestCase):
    def test_fit(self):
        gmr = GaussianMLPRegressorWithModel(input_shape=(1, ), output_dim=1)
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{'observations': [[x]], 'returns': [np.sin(x)]} for x in data]

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        for _ in range(100):
            gmr.fit(observations, returns.reshape((-1, 1)))

        paths = {
            'observations': [[-np.pi], [-np.pi / 2], [-np.pi / 4], [0],
                             [np.pi / 4], [np.pi / 2], [np.pi]]
        }

        prediction = gmr.predict(paths['observations'])

        expected = [[0], [-1], [-0.707], [0], [0.707], [1], [0]]
        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

    def test_fit_custom(self):
        gmr = GaussianMLPRegressorWithModel(
            input_shape=(1, ),
            output_dim=1,
            subsample_factor=0.9,
            normalize_inputs=False,
            normalize_outputs=False)
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{'observations': [[x]], 'returns': [np.sin(x)]} for x in data]

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        for _ in range(100):
            gmr.fit(observations, returns.reshape((-1, 1)))

        paths = {
            'observations': [[-np.pi], [-np.pi / 2], [-np.pi / 4], [0],
                             [np.pi / 4], [np.pi / 2], [np.pi]]
        }

        prediction = gmr.predict(paths['observations'])

        expected = [[0], [-1], [-0.707], [0], [0.707], [1], [0]]
        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

    def test_fit_without_trusted_region(self):
        gmr = GaussianMLPRegressorWithModel(
            input_shape=(1, ), output_dim=1, use_trust_region=False)
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{'observations': [[x]], 'returns': [np.sin(x)]} for x in data]

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        for _ in range(100):
            gmr.fit(observations, returns.reshape((-1, 1)))

        paths = {
            'observations': [[-np.pi], [-np.pi / 2], [-np.pi / 4], [0],
                             [np.pi / 4], [np.pi / 2], [np.pi]]
        }

        prediction = gmr.predict(paths['observations'])

        expected = [[0], [-1], [-0.707], [0], [0.707], [1], [0]]
        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

    @params((1, (1, )), (1, (2, )), (2, (3, )), (2, (1, 1)), (3, (2, 2)))
    def test_log_likelihood_sym(self, output_dim, input_shape):
        gmr = GaussianMLPRegressorWithModel(
            input_shape=input_shape,
            output_dim=output_dim,
            optimizer=PenaltyLbfgsOptimizer,
            optimizer_args=dict())

        new_input_var = tf.placeholder(
            tf.float32, shape=(None, ) + input_shape)
        new_ys_var = tf.placeholder(
            dtype=tf.float32, name='ys', shape=(None, output_dim))

        data = np.random.random(size=input_shape)
        label = np.ones(output_dim)

        outputs = gmr.log_likelihood_sym(
            new_input_var, new_ys_var, name='ll_sym')
        ll_from_sym = self.sess.run(
            outputs, feed_dict={
                new_input_var: [data],
                new_ys_var: [label]
            })
        mean, log_std = gmr._f_pdists([data])
        ll = gmr.model.networks['default'].dist.log_likelihood(
            [label], dict(mean=mean, log_std=log_std))
        assert np.allclose(ll, ll_from_sym, rtol=0, atol=1e-5)
