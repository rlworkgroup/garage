import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.optimizers import ConjugateGradientOptimizer, LbfgsOptimizer
from garage.tf.regressors import BernoulliMLPRegressor
from tests.fixtures import TfGraphTestCase


def get_labels(input_shape, xs, output_dim):
    if input_shape == (1, ):
        label = [0, 0]
        # [0, 1] if sign is positive else [1, 0]
        ys = 0 if np.sin(xs) <= 0 else 1
        label[ys] = 1
    elif input_shape == (2, ):
        ys = int(np.round(xs[0])) ^ int(np.round(xs[1]))
        if output_dim == 1:
            label = ys
        else:
            # [0, 1] if XOR is 1 else [1, 0]
            label = [0, 0]
            label[ys] = 1
    return label


def get_train_data(input_shape, output_dim):
    if input_shape == (1, ):
        # Sign of sin function
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{
            'observations': [[x]],
            'returns': [get_labels(input_shape, x, output_dim)]
        } for x in data]
    elif input_shape == (2, ):
        # Generate 1000 points with coordinates in [0, 1] for XOR data
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 10)
        data = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
        obs = [{
            'observations': [x],
            'returns': [get_labels(input_shape, x, output_dim)]
        } for x in data]
    observations = np.concatenate([p['observations'] for p in obs])
    returns = np.concatenate([p['returns'] for p in obs])
    returns = returns.reshape((-1, output_dim))
    return observations, returns


def get_test_data(input_shape, output_dim):
    if input_shape == (1, ):
        paths = {
            'observations': [[-np.pi / 2], [-np.pi / 3], [-np.pi / 4],
                             [np.pi / 4], [np.pi / 3], [np.pi / 4]]
        }
        expected = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]
    elif input_shape == (2, ):
        paths = {'observations': [[0, 0], [0, 1], [1, 0], [1, 1]]}
        if output_dim == 1:
            expected = [[0], [1], [1], [0]]
        else:
            expected = [[1, 0], [0, 1], [0, 1], [1, 0]]
    return paths, expected


class TestBernoulliMLPRegressor(TfGraphTestCase):
    # yapf: disable
    @pytest.mark.parametrize('input_shape, output_dim', [
        ((1, ), 2),
        ((2, ), 1),
        ((2, ), 2),
    ])
    # yapf: enable
    def test_fit_normalized(self, input_shape, output_dim):
        bmr = BernoulliMLPRegressor(input_shape=input_shape,
                                    output_dim=output_dim)

        observations, returns = get_train_data(input_shape, output_dim)

        for _ in range(150):
            bmr.fit(observations, returns)

        paths, expected = get_test_data(input_shape, output_dim)

        prediction = np.cast['int'](bmr.predict(paths['observations']))
        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(bmr.model.networks['default'].x_mean)
        x_mean_expected = np.mean(observations, axis=0, keepdims=True)
        x_std = self.sess.run(bmr.model.networks['default'].x_std)
        x_std_expected = np.std(observations, axis=0, keepdims=True)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

    # yapf: disable
    @pytest.mark.parametrize('input_shape, output_dim', [
        ((1, ), 2),
        ((2, ), 2),
        ((2, ), 1),
    ])
    # yapf: enable
    def test_fit_unnormalized(self, input_shape, output_dim):
        bmr = BernoulliMLPRegressor(input_shape=input_shape,
                                    output_dim=output_dim,
                                    normalize_inputs=False)

        observations, returns = get_train_data(input_shape, output_dim)

        for _ in range(150):
            bmr.fit(observations, returns)

        paths, expected = get_test_data(input_shape, output_dim)

        prediction = np.cast['int'](bmr.predict(paths['observations']))

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(bmr.model.networks['default'].x_mean)
        x_mean_expected = np.zeros_like(x_mean)
        x_std = self.sess.run(bmr.model.networks['default'].x_std)
        x_std_expected = np.ones_like(x_std)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

    # yapf: disable
    @pytest.mark.parametrize('input_shape, output_dim', [
        ((1, ), 2),
        ((2, ), 2),
        ((2, ), 1),
    ])
    # yapf: enable
    def test_fit_with_no_trust_region(self, input_shape, output_dim):
        bmr = BernoulliMLPRegressor(input_shape=input_shape,
                                    output_dim=output_dim,
                                    use_trust_region=False)

        observations, returns = get_train_data(input_shape, output_dim)

        for _ in range(150):
            bmr.fit(observations, returns)

        paths, expected = get_test_data(input_shape, output_dim)
        prediction = np.cast['int'](bmr.predict(paths['observations']))

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(bmr.model.networks['default'].x_mean)
        x_mean_expected = np.mean(observations, axis=0, keepdims=True)
        x_std = self.sess.run(bmr.model.networks['default'].x_std)
        x_std_expected = np.std(observations, axis=0, keepdims=True)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

    def test_sample_predict(self):
        n_sample = 100
        input_dim = 50
        output_dim = 1
        bmr = BernoulliMLPRegressor(input_shape=(input_dim, ),
                                    output_dim=output_dim)

        xs = np.random.random((input_dim, ))
        p = bmr._f_prob([xs])
        ys = bmr.sample_predict([xs] * n_sample)
        p_predict = np.count_nonzero(ys == 1) / n_sample

        assert np.real_if_close(p, p_predict)

    def test_predict_log_likelihood(self):
        n_sample = 50
        input_dim = 50
        output_dim = 1
        bmr = BernoulliMLPRegressor(input_shape=(input_dim, ),
                                    output_dim=output_dim)

        xs = np.random.random((n_sample, input_dim))
        ys = np.random.randint(2, size=(n_sample, output_dim))
        p = bmr._f_prob(xs)
        ll = bmr.predict_log_likelihood(xs, ys)
        ll_true = np.sum(np.log(p * ys + (1 - p) * (1 - ys)), axis=-1)

        assert np.allclose(ll, ll_true)

    # yapf: disable
    @pytest.mark.parametrize('output_dim, input_shape', [
        (1, (1, 1)),
        (1, (2, 2)),
        (2, (3, 2)),
        (3, (2, 2)),
    ])
    # yapf: enable
    def test_log_likelihood_sym(self, output_dim, input_shape):
        bmr = BernoulliMLPRegressor(input_shape=(input_shape[1], ),
                                    output_dim=output_dim)

        new_xs_var = tf.compat.v1.placeholder(tf.float32, input_shape)
        new_ys_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                              name='ys',
                                              shape=(None, output_dim))

        data = np.full(input_shape, 0.5)
        one_hot_label = np.zeros((input_shape[0], output_dim))
        one_hot_label[np.arange(input_shape[0]), 0] = 1

        p = bmr._f_prob(np.asarray(data))
        ll = bmr._dist.log_likelihood(np.asarray(one_hot_label), dict(p=p))

        outputs = bmr.log_likelihood_sym(new_xs_var, new_ys_var, name='ll_sym')

        ll_from_sym = self.sess.run(outputs,
                                    feed_dict={
                                        new_xs_var: data,
                                        new_ys_var: one_hot_label
                                    })

        assert np.allclose(ll, ll_from_sym, rtol=0, atol=1e-5)

    @mock.patch('tests.garage.tf.regressors.'
                'test_bernoulli_mlp_regressor.'
                'LbfgsOptimizer')
    @mock.patch('tests.garage.tf.regressors.'
                'test_bernoulli_mlp_regressor.'
                'ConjugateGradientOptimizer')
    def test_optimizer_args(self, mock_cg, mock_lbfgs):
        lbfgs_args = dict(max_opt_itr=25)
        cg_args = dict(cg_iters=15)
        bmr = BernoulliMLPRegressor(input_shape=(1, ),
                                    output_dim=2,
                                    optimizer=LbfgsOptimizer,
                                    optimizer_args=lbfgs_args,
                                    tr_optimizer=ConjugateGradientOptimizer,
                                    tr_optimizer_args=cg_args,
                                    use_trust_region=True)

        assert mock_lbfgs.return_value is bmr._optimizer
        assert mock_cg.return_value is bmr._tr_optimizer

        mock_lbfgs.assert_called_with(max_opt_itr=25)
        mock_cg.assert_called_with(cg_iters=15)

    def test_is_pickleable(self):
        bmr = BernoulliMLPRegressor(input_shape=(1, ), output_dim=2)

        with tf.compat.v1.variable_scope(
                'BernoulliMLPRegressor/NormalizedInputMLPModel', reuse=True):
            bias = tf.compat.v1.get_variable('mlp/hidden_0/bias')
        bias.load(tf.ones_like(bias).eval())
        bias1 = bias.eval()

        result1 = np.cast['int'](bmr.predict(np.ones((1, 1))))
        h = pickle.dumps(bmr)

        with tf.compat.v1.Session(graph=tf.Graph()):
            bmr_pickled = pickle.loads(h)
            result2 = np.cast['int'](bmr_pickled.predict(np.ones((1, 1))))
            assert np.array_equal(result1, result2)

            with tf.compat.v1.variable_scope(
                    'BernoulliMLPRegressor/NormalizedInputMLPModel',
                    reuse=True):
                bias2 = tf.compat.v1.get_variable('mlp/hidden_0/bias').eval()

            assert np.array_equal(bias1, bias2)

    def test_is_pickleable2(self):
        bmr = BernoulliMLPRegressor(input_shape=(1, ), output_dim=2)

        with tf.compat.v1.variable_scope(
                'BernoulliMLPRegressor/NormalizedInputMLPModel', reuse=True):
            x_mean = tf.compat.v1.get_variable('normalized_vars/x_mean')
        x_mean.load(tf.ones_like(x_mean).eval())
        x1 = bmr.model.networks['default'].x_mean.eval()
        h = pickle.dumps(bmr)
        with tf.compat.v1.Session(graph=tf.Graph()):
            bmr_pickled = pickle.loads(h)
            x2 = bmr_pickled.model.networks['default'].x_mean.eval()
            assert np.array_equal(x1, x2)
