import pickle

from nose2.tools.params import params
import numpy as np
import tensorflow as tf

from garage.tf.regressors import CategoricalMLPRegressorWithModel
from tests.fixtures import TfGraphTestCase


def get_labels(input):
    label = [0, 0]
    label[0 if np.sin(input) <= 0 else 1] = 1
    return label


class TestCategoricalMLPRegressorWithModel(TfGraphTestCase):
    def test_fit_normalized(self):
        cmr = CategoricalMLPRegressorWithModel(input_shape=(1, ), output_dim=2)
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{
            'observations': [[x]],
            'returns': [get_labels(x)]
        } for x in data]

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        returns = returns.reshape((-1, 2))

        for _ in range(150):
            cmr.fit(observations, returns)

        paths = {
            'observations': [[-np.pi / 2], [-np.pi / 3], [-np.pi / 4],
                             [np.pi / 4], [np.pi / 3], [np.pi / 4]]
        }

        prediction = cmr.predict(paths['observations'])

        expected = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(cmr.model.networks['default'].x_mean)
        x_mean_expected = np.mean(observations, axis=0, keepdims=True)
        x_std = self.sess.run(cmr.model.networks['default'].x_std)
        x_std_expected = np.std(observations, axis=0, keepdims=True)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

    def test_fit_unnormalized(self):
        cmr = CategoricalMLPRegressorWithModel(
            input_shape=(1, ), output_dim=2, normalize_inputs=False)
        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{
            'observations': [[x]],
            'returns': [get_labels(x)]
        } for x in data]

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        returns = returns.reshape((-1, 2))

        for _ in range(150):
            cmr.fit(observations, returns)

        paths = {
            'observations': [[-np.pi / 2], [-np.pi / 3], [-np.pi / 4],
                             [np.pi / 4], [np.pi / 3], [np.pi / 4]]
        }

        prediction = cmr.predict(paths['observations'])

        expected = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(cmr.model.networks['default'].x_mean)
        x_mean_expected = np.zeros_like(x_mean)
        x_std = self.sess.run(cmr.model.networks['default'].x_std)
        x_std_expected = np.ones_like(x_std)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

    def test_fit_with_initial_trust_region(self):
        cmr = CategoricalMLPRegressorWithModel(
            input_shape=(1, ), output_dim=2, no_initial_trust_region=False)

        data = np.linspace(-np.pi, np.pi, 1000)
        obs = [{
            'observations': [[x]],
            'returns': [get_labels(x)]
        } for x in data]

        observations = np.concatenate([p['observations'] for p in obs])
        returns = np.concatenate([p['returns'] for p in obs])
        returns = returns.reshape((-1, 2))

        for _ in range(150):
            cmr.fit(observations, returns)

        paths = {
            'observations': [[-np.pi / 2], [-np.pi / 3], [-np.pi / 4],
                             [np.pi / 4], [np.pi / 3], [np.pi / 4]]
        }

        prediction = cmr.predict(paths['observations'])

        expected = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(cmr.model.networks['default'].x_mean)
        x_mean_expected = np.mean(observations, axis=0, keepdims=True)
        x_std = self.sess.run(cmr.model.networks['default'].x_std)
        x_std_expected = np.std(observations, axis=0, keepdims=True)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

    @params((1, (1, )), (1, (2, )), (2, (3, )), (2, (1, 1)), (3, (2, 2)))
    def test_dist_info_sym(self, output_dim, input_shape):
        cmr = CategoricalMLPRegressorWithModel(
            input_shape=input_shape, output_dim=output_dim)

        new_xs_var = tf.placeholder(tf.float32, shape=(None, ) + input_shape)

        data = np.random.random(size=input_shape)
        label = np.random.randint(0, output_dim)
        one_hot_label = np.zeros(output_dim)
        one_hot_label[label] = 1

        outputs = cmr.dist_info_sym(new_xs_var, name='dist_info_sym')
        prob = self.sess.run(outputs, feed_dict={new_xs_var: [data]})

        expected_prob = cmr._f_prob(np.asarray([data]))

        assert np.allclose(prob['prob'], expected_prob, rtol=0, atol=1e-5)

    @params((1, (1, )), (1, (2, )), (2, (3, )), (2, (1, 1)), (3, (2, 2)))
    def test_log_likelihood_sym(self, output_dim, input_shape):
        cmr = CategoricalMLPRegressorWithModel(
            input_shape=input_shape, output_dim=output_dim)

        new_xs_var = tf.placeholder(tf.float32, shape=(None, ) + input_shape)
        new_ys_var = tf.placeholder(
            dtype=tf.float32, name='ys', shape=(None, output_dim))

        data = np.random.random(size=input_shape)
        label = np.random.randint(0, output_dim)
        one_hot_label = np.zeros(output_dim)
        one_hot_label[label] = 1

        ll = cmr.predict_log_likelihood([data], [one_hot_label])

        outputs = cmr.log_likelihood_sym(new_xs_var, new_ys_var, name='ll_sym')

        ll_from_sym = self.sess.run(
            outputs,
            feed_dict={
                new_xs_var: [data],
                new_ys_var: [one_hot_label]
            })

        assert np.allclose(ll, ll_from_sym, rtol=0, atol=1e-5)

    def test_is_pickleable(self):
        cmr = CategoricalMLPRegressorWithModel(input_shape=(1, ), output_dim=2)

        with tf.variable_scope(
                'CategoricalMLPRegressorWithModel/NormalizedInputMLPModel',
                reuse=True):
            bias = tf.get_variable('mlp/hidden_0/bias')
        bias.load(tf.ones_like(bias).eval())

        result1 = cmr.predict(np.ones((1, 1)))

        h = pickle.dumps(cmr)
        with tf.Session(graph=tf.Graph()):
            cmr_pickled = pickle.loads(h)
            result2 = cmr_pickled.predict(np.ones((1, 1)))
            assert np.array_equal(result1, result2)

    def test_is_pickleable2(self):
        cmr = CategoricalMLPRegressorWithModel(input_shape=(1, ), output_dim=2)

        with tf.variable_scope(
                'CategoricalMLPRegressorWithModel/NormalizedInputMLPModel',
                reuse=True):
            x_mean = tf.get_variable('normalized_vars/x_mean')
        x_mean.load(tf.ones_like(x_mean).eval())
        x1 = cmr.model.networks['default'].x_mean.eval()
        h = pickle.dumps(cmr)
        with tf.Session(graph=tf.Graph()):
            cmr_pickled = pickle.loads(h)
            x2 = cmr_pickled.model.networks['default'].x_mean.eval()
            assert np.array_equal(x1, x2)
