import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from garage.tf.optimizers import ConjugateGradientOptimizer, LbfgsOptimizer
from garage.tf.regressors import CategoricalMLPRegressor
from tests.fixtures import TfGraphTestCase


def get_labels(input_shape, xs):
    label = [0, 0]
    if input_shape == (1, ):
        ys = 0 if np.sin(xs) <= 0 else 1
        label[ys] = 1

    elif input_shape == (2, ):
        ys = int(np.round(xs[0])) ^ int(np.round(xs[1]))
        label[ys] = 1

    return label


def get_train_data(input_shape):
    if input_shape == (1, ):
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{
            'observations': [[x]],
            'returns': [get_labels(input_shape, x)]
        } for x in data]

    elif input_shape == (2, ):
        data = [np.random.rand(2) for _ in range(1000)]
        obs = [{
            'observations': [x],
            'returns': [get_labels(input_shape, x)]
        } for x in data]
    return obs


def get_test_data(input_shape):
    if input_shape == (1, ):
        paths = {
            'observations': [[-np.pi / 2], [-np.pi / 3], [-np.pi / 4],
                             [np.pi / 4], [np.pi / 3], [np.pi / 4]]
        }
        expected = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]

    elif input_shape == (2, ):
        paths = {'observations': [[0, 0], [0, 1], [1, 0], [1, 1]]}
        expected = [[1, 0], [0, 1], [0, 1], [1, 0]]

    return paths, expected


class TestCategoricalMLPRegressor(TfGraphTestCase):

    def test_dist(self):
        cmr = CategoricalMLPRegressor(input_shape=(1, ), output_dim=2)
        dist = cmr._network.dist
        assert isinstance(dist, tfp.distributions.OneHotCategorical)

    @pytest.mark.parametrize('input_shape, output_dim', [((1, ), 2),
                                                         ((2, ), 2)])
    def test_fit_normalized(self, input_shape, output_dim):
        cmr = CategoricalMLPRegressor(input_shape=input_shape,
                                      output_dim=output_dim)
        obs = get_train_data(input_shape)

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        returns = returns.reshape((-1, 2))

        for _ in range(150):
            cmr.fit(observations, returns)

        paths, expected = get_test_data(input_shape)

        prediction = cmr.predict(paths['observations'])

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(cmr._network.x_mean)
        x_mean_expected = np.mean(observations, axis=0, keepdims=True)
        x_std = self.sess.run(cmr._network.x_std)
        x_std_expected = np.std(observations, axis=0, keepdims=True)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

    @pytest.mark.parametrize('input_shape, output_dim', [((1, ), 2),
                                                         ((2, ), 2)])
    def test_fit_unnormalized(self, input_shape, output_dim):
        cmr = CategoricalMLPRegressor(input_shape=input_shape,
                                      output_dim=output_dim,
                                      normalize_inputs=False)
        obs = get_train_data(input_shape)

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        returns = returns.reshape((-1, 2))

        for _ in range(150):
            cmr.fit(observations, returns)

        paths, expected = get_test_data(input_shape)

        prediction = cmr.predict(paths['observations'])

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(cmr._network.x_mean)
        x_mean_expected = np.zeros_like(x_mean)
        x_std = self.sess.run(cmr._network.x_std)
        x_std_expected = np.ones_like(x_std)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

    @pytest.mark.parametrize('input_shape, output_dim', [((1, ), 2),
                                                         ((2, ), 2)])
    def test_fit_without_initial_trust_region(self, input_shape, output_dim):
        cmr = CategoricalMLPRegressor(input_shape=input_shape,
                                      output_dim=output_dim,
                                      use_trust_region=False)
        obs = get_train_data(input_shape)

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        returns = returns.reshape((-1, 2))

        for _ in range(150):
            cmr.fit(observations, returns)

        paths, expected = get_test_data(input_shape)

        prediction = cmr.predict(paths['observations'])

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(cmr._network.x_mean)
        x_mean_expected = np.mean(observations, axis=0, keepdims=True)
        x_std = self.sess.run(cmr._network.x_std)
        x_std_expected = np.std(observations, axis=0, keepdims=True)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

    @mock.patch('tests.garage.tf.regressors.'
                'test_categorical_mlp_regressor.'
                'LbfgsOptimizer')
    @mock.patch('tests.garage.tf.regressors.'
                'test_categorical_mlp_regressor.'
                'ConjugateGradientOptimizer')
    def test_optimizer_args(self, mock_cg, mock_lbfgs):
        lbfgs_args = dict(max_opt_itr=25)
        cg_args = dict(cg_iters=15)
        cmr = CategoricalMLPRegressor(input_shape=(1, ),
                                      output_dim=2,
                                      optimizer=LbfgsOptimizer,
                                      optimizer_args=lbfgs_args,
                                      tr_optimizer=ConjugateGradientOptimizer,
                                      tr_optimizer_args=cg_args,
                                      use_trust_region=True)

        assert mock_lbfgs.return_value is cmr._optimizer
        assert mock_cg.return_value is cmr._tr_optimizer

        mock_lbfgs.assert_called_with(max_opt_itr=25)
        mock_cg.assert_called_with(cg_iters=15)

    def test_is_pickleable(self):
        cmr = CategoricalMLPRegressor(input_shape=(1, ), output_dim=2)

        with tf.compat.v1.variable_scope(
                'CategoricalMLPRegressor/CategoricalMLPRegressorModel',
                reuse=True):
            bias = tf.compat.v1.get_variable('mlp/hidden_0/bias')
        bias.load(tf.ones_like(bias).eval())

        result1 = cmr.predict(np.ones((1, 1)))

        h = pickle.dumps(cmr)
        with tf.compat.v1.Session(graph=tf.Graph()):
            cmr_pickled = pickle.loads(h)
            result2 = cmr_pickled.predict(np.ones((1, 1)))
            assert np.array_equal(result1, result2)

    def test_is_pickleable2(self):
        cmr = CategoricalMLPRegressor(input_shape=(1, ), output_dim=2)

        with tf.compat.v1.variable_scope(
                'CategoricalMLPRegressor/CategoricalMLPRegressorModel',
                reuse=True):
            x_mean = tf.compat.v1.get_variable('normalized_vars/x_mean')
        x_mean.load(tf.ones_like(x_mean).eval())
        x1 = cmr.model._networks['default'].x_mean.eval()
        h = pickle.dumps(cmr)
        with tf.compat.v1.Session(graph=tf.Graph()):
            cmr_pickled = pickle.loads(h)
            x2 = cmr_pickled.model._networks['default'].x_mean.eval()
            assert np.array_equal(x1, x2)
