from nose2.tools.params import params
import numpy as np
import tensorflow as tf

from garage.tf.optimizers import LbfgsOptimizer
from garage.tf.regressors import DeterministicMLPRegressorWithModel
from tests.fixtures import TfGraphTestCase


class TestDeterministicMLPRegressorWithModel(TfGraphTestCase):
    def test_fit_normalized(self):
        gmr = DeterministicMLPRegressorWithModel(
            input_shape=(1, ), output_dim=1)
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

    def test_fit_unnormalized(self):
        gmr = DeterministicMLPRegressorWithModel(
            input_shape=(1, ), output_dim=1, normalize_inputs=False)
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

    @params((1, (1, )), (1, (2, )), (2, (3, )), (2, (1, 1)), (3, (2, 2)))
    def test_predict_sym(self, output_dim, input_shape):
        gmr = DeterministicMLPRegressorWithModel(
            input_shape=input_shape,
            output_dim=output_dim,
            optimizer=LbfgsOptimizer,
            optimizer_args=dict())

        new_input_var = tf.placeholder(
            tf.float32, shape=(None, ) + input_shape)

        data = np.random.random(size=input_shape)

        outputs = gmr.predict_sym(new_input_var, name='y_hat_sym')
        y_hat_sym = self.sess.run(outputs, feed_dict={new_input_var: [data]})
        y_hat = gmr._f_predict([data])
        assert np.allclose(y_hat, y_hat_sym, rtol=0, atol=1e-5)
