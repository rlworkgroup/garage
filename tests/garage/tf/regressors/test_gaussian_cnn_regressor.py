import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.optimizers import LbfgsOptimizer
from garage.tf.regressors import GaussianCNNRegressor
from tests.fixtures import TfGraphTestCase


def get_train_test_data():
    matrices = [
        np.linspace(i - 0.5, i + 0.5, 300).reshape((10, 10, 3))
        for i in range(110)
    ]
    data = [np.sin(matrices[i]) for i in range(100)]
    obs = [{'observations': [x], 'returns': [np.mean(x)]} for x in data]

    observations = np.concatenate([p['observations'] for p in obs])
    returns = np.concatenate([p['returns'] for p in obs])
    returns = returns.reshape((-1, 1))

    paths = {'observations': [np.sin(matrices[i]) for i in range(100, 110)]}

    expected = [[np.mean(x)] for x in paths['observations']]

    return (observations, returns), (paths, expected)


class TestGaussianCNNRegressor(TfGraphTestCase):

    @pytest.mark.large
    def test_fit_normalized(self):
        gcr = GaussianCNNRegressor(input_shape=(10, 10, 3),
                                   num_filters=(3, 6),
                                   filter_dims=(3, 3),
                                   strides=(1, 1),
                                   padding='SAME',
                                   hidden_sizes=(32, ),
                                   output_dim=1,
                                   adaptive_std=False,
                                   use_trust_region=True)

        train_data, test_data = get_train_test_data()
        observations, returns = train_data

        for _ in range(20):
            gcr.fit(observations, returns)

        paths, expected = test_data

        prediction = gcr.predict(paths['observations'])
        average_error = 0.0
        for i in range(len(expected)):
            average_error += np.abs(expected[i] - prediction[i])
        average_error /= len(expected)
        assert average_error <= 0.05

        x_mean = self.sess.run(gcr.model.networks['default'].x_mean)
        x_mean_expected = np.mean(observations, axis=0, keepdims=True)
        x_std = self.sess.run(gcr.model.networks['default'].x_std)
        x_std_expected = np.std(observations, axis=0, keepdims=True)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

        y_mean = self.sess.run(gcr.model.networks['default'].y_mean)
        y_mean_expected = np.mean(returns, axis=0, keepdims=True)
        y_std = self.sess.run(gcr.model.networks['default'].y_std)
        y_std_expected = np.std(returns, axis=0, keepdims=True)

        assert np.allclose(y_mean, y_mean_expected)
        assert np.allclose(y_std, y_std_expected)

    @pytest.mark.large
    def test_fit_unnormalized(self):
        gcr = GaussianCNNRegressor(input_shape=(10, 10, 3),
                                   num_filters=(3, 6),
                                   filter_dims=(3, 3),
                                   strides=(1, 1),
                                   padding='SAME',
                                   hidden_sizes=(32, ),
                                   output_dim=1,
                                   adaptive_std=True,
                                   normalize_inputs=False,
                                   normalize_outputs=False)

        train_data, test_data = get_train_test_data()
        observations, returns = train_data

        for _ in range(20):
            gcr.fit(observations, returns)

        paths, expected = test_data

        prediction = gcr.predict(paths['observations'])
        average_error = 0.0
        for i in range(len(expected)):
            average_error += np.abs(expected[i] - prediction[i])
        average_error /= len(expected)
        assert average_error <= 0.1

        x_mean = self.sess.run(gcr.model.networks['default'].x_mean)
        x_mean_expected = np.zeros_like(x_mean)
        x_std = self.sess.run(gcr.model.networks['default'].x_std)
        x_std_expected = np.ones_like(x_std)
        assert np.array_equal(x_mean, x_mean_expected)
        assert np.array_equal(x_std, x_std_expected)

        y_mean = self.sess.run(gcr.model.networks['default'].y_mean)
        y_mean_expected = np.zeros_like(y_mean)
        y_std = self.sess.run(gcr.model.networks['default'].y_std)
        y_std_expected = np.ones_like(y_std)

        assert np.allclose(y_mean, y_mean_expected)
        assert np.allclose(y_std, y_std_expected)

    @pytest.mark.large
    def test_fit_smaller_subsample_factor(self):
        gcr = GaussianCNNRegressor(input_shape=(10, 10, 3),
                                   num_filters=(3, 6),
                                   filter_dims=(3, 3),
                                   strides=(1, 1),
                                   padding='SAME',
                                   hidden_sizes=(32, ),
                                   output_dim=1,
                                   subsample_factor=0.9,
                                   adaptive_std=False)
        train_data, test_data = get_train_test_data()
        observations, returns = train_data

        for _ in range(20):
            gcr.fit(observations, returns)

        paths, expected = test_data

        prediction = gcr.predict(paths['observations'])
        average_error = 0.0
        for i in range(len(expected)):
            average_error += np.abs(expected[i] - prediction[i])
        average_error /= len(expected)
        assert average_error <= 0.05

    @pytest.mark.large
    def test_fit_without_trusted_region(self):
        gcr = GaussianCNNRegressor(input_shape=(10, 10, 3),
                                   num_filters=(3, 6),
                                   filter_dims=(3, 3),
                                   strides=(1, 1),
                                   padding='SAME',
                                   hidden_sizes=(32, ),
                                   output_dim=1,
                                   adaptive_std=False,
                                   use_trust_region=False)
        train_data, test_data = get_train_test_data()
        observations, returns = train_data

        for _ in range(20):
            gcr.fit(observations, returns)

        paths, expected = test_data

        prediction = gcr.predict(paths['observations'])
        average_error = 0.0
        for i in range(len(expected)):
            average_error += np.abs(expected[i] - prediction[i])
        average_error /= len(expected)
        assert average_error <= 0.05

    @pytest.mark.parametrize('output_dim', [(1), (2), (3)])
    def test_log_likelihood_sym(self, output_dim):
        input_shape = (28, 28, 3)
        gcr = GaussianCNNRegressor(input_shape=input_shape,
                                   num_filters=(3, 6),
                                   filter_dims=(3, 3),
                                   strides=(1, 1),
                                   padding='SAME',
                                   hidden_sizes=(32, ),
                                   output_dim=1,
                                   adaptive_std=False,
                                   use_trust_region=False)

        new_input_var = tf.compat.v1.placeholder(tf.float32,
                                                 shape=(None, ) + input_shape)
        new_ys_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                              name='ys',
                                              shape=(None, output_dim))

        data = np.full(input_shape, 0.5)
        label = np.ones(output_dim)

        outputs = gcr.log_likelihood_sym(new_input_var,
                                         new_ys_var,
                                         name='ll_sym')
        ll_from_sym = self.sess.run(outputs,
                                    feed_dict={
                                        new_input_var: [data],
                                        new_ys_var: [label]
                                    })
        mean, log_std = gcr._f_pdists([data])
        ll = gcr.model.networks['default'].dist.log_likelihood(
            [label], dict(mean=mean, log_std=log_std))
        assert np.allclose(ll, ll_from_sym, rtol=0, atol=1e-5)

    @mock.patch('tests.garage.tf.regressors.'
                'test_gaussian_cnn_regressor.'
                'LbfgsOptimizer')
    def test_optimizer_args(self, mock_lbfgs):
        lbfgs_args = dict(max_opt_itr=25)
        gcr = GaussianCNNRegressor(input_shape=(10, 10, 3),
                                   num_filters=(3, 6),
                                   filter_dims=(3, 3),
                                   strides=(1, 1),
                                   padding='SAME',
                                   hidden_sizes=(32, ),
                                   output_dim=1,
                                   optimizer=LbfgsOptimizer,
                                   optimizer_args=lbfgs_args,
                                   use_trust_region=True)

        assert mock_lbfgs.return_value is gcr._optimizer

        mock_lbfgs.assert_called_with(max_opt_itr=25)

    def test_is_pickleable(self):
        input_shape = (28, 28, 3)
        gcr = GaussianCNNRegressor(input_shape=input_shape,
                                   num_filters=(3, 6),
                                   filter_dims=(3, 3),
                                   strides=(1, 1),
                                   padding='SAME',
                                   hidden_sizes=(32, ),
                                   output_dim=1,
                                   adaptive_std=False,
                                   use_trust_region=False)

        with tf.compat.v1.variable_scope(
                'GaussianCNNRegressor/GaussianCNNRegressorModel', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/hidden_0/bias')
        bias.load(tf.ones_like(bias).eval())

        result1 = gcr.predict([np.ones(input_shape)])
        h = pickle.dumps(gcr)

        with tf.compat.v1.Session(graph=tf.Graph()):
            gcr_pickled = pickle.loads(h)
            result2 = gcr_pickled.predict([np.ones(input_shape)])
            assert np.array_equal(result1, result2)

    def test_is_pickleable2(self):
        input_shape = (28, 28, 3)
        gcr = GaussianCNNRegressor(input_shape=input_shape,
                                   num_filters=(3, 6),
                                   filter_dims=(3, 3),
                                   strides=(1, 1),
                                   padding='SAME',
                                   hidden_sizes=(32, ),
                                   output_dim=1,
                                   adaptive_std=False,
                                   use_trust_region=False)

        with tf.compat.v1.variable_scope(
                'GaussianCNNRegressor/GaussianCNNRegressorModel', reuse=True):
            x_mean = tf.compat.v1.get_variable('normalized_vars/x_mean')
        x_mean.load(tf.ones_like(x_mean).eval())
        x1 = gcr.model.networks['default'].x_mean.eval()
        h = pickle.dumps(gcr)
        with tf.compat.v1.Session(graph=tf.Graph()):
            gcr_pickled = pickle.loads(h)
            x2 = gcr_pickled.model.networks['default'].x_mean.eval()
            assert np.array_equal(x1, x2)
