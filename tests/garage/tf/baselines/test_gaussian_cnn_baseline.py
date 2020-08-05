import pickle
from unittest import mock

import akro
import numpy as np
import pytest
import tensorflow as tf

from garage import EnvSpec
from garage.envs import GymEnv
from garage.tf.baselines import GaussianCNNBaseline
from garage.tf.optimizers import LbfgsOptimizer

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


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

    return (obs, observations, returns), (paths, expected)


test_env_spec = EnvSpec(observation_space=akro.Box(low=-1,
                                                   high=1,
                                                   shape=(10, 10, 3)),
                        action_space=None)


class TestGaussianCNNBaseline(TfGraphTestCase):

    @pytest.mark.large
    def test_fit_normalized(self):
        gcr = GaussianCNNBaseline(env_spec=test_env_spec,
                                  filters=((3, (3, 3)), (6, (3, 3))),
                                  strides=(1, 1),
                                  padding='SAME',
                                  hidden_sizes=(32, ),
                                  adaptive_std=False,
                                  use_trust_region=True)

        train_data, test_data = get_train_test_data()
        train_paths, observations, returns = train_data
        for _ in range(30):
            gcr.fit(train_paths)

        test_paths, expected = test_data
        prediction = gcr.predict(test_paths)

        average_error = 0.0
        for i, exp in enumerate(expected):
            average_error += np.abs(exp - prediction[i])
        average_error /= len(expected)
        assert average_error <= 0.1

        x_mean = self.sess.run(gcr._networks['default'].x_mean)
        x_mean_expected = np.mean(observations, axis=0, keepdims=True)
        x_std = self.sess.run(gcr._networks['default'].x_std)
        x_std_expected = np.std(observations, axis=0, keepdims=True)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

        y_mean = self.sess.run(gcr._networks['default'].y_mean)
        y_mean_expected = np.mean(returns, axis=0, keepdims=True)
        y_std = self.sess.run(gcr._networks['default'].y_std)
        y_std_expected = np.std(returns, axis=0, keepdims=True)

        assert np.allclose(y_mean, y_mean_expected)
        assert np.allclose(y_std, y_std_expected)

    @pytest.mark.large
    def test_fit_unnormalized(self):
        gcr = GaussianCNNBaseline(env_spec=test_env_spec,
                                  filters=((3, (3, 3)), (6, (3, 3))),
                                  strides=(1, 1),
                                  padding='SAME',
                                  hidden_sizes=(32, ),
                                  adaptive_std=True,
                                  normalize_inputs=False,
                                  normalize_outputs=False)

        train_data, test_data = get_train_test_data()
        train_paths, _, _ = train_data

        for _ in range(30):
            gcr.fit(train_paths)

        test_paths, expected = test_data

        prediction = gcr.predict(test_paths)
        average_error = 0.0
        for i, exp in enumerate(expected):
            average_error += np.abs(exp - prediction[i])
        average_error /= len(expected)
        assert average_error <= 0.1

        x_mean = self.sess.run(gcr._networks['default'].x_mean)
        x_mean_expected = np.zeros_like(x_mean)
        x_std = self.sess.run(gcr._networks['default'].x_std)
        x_std_expected = np.ones_like(x_std)
        assert np.array_equal(x_mean, x_mean_expected)
        assert np.array_equal(x_std, x_std_expected)

        y_mean = self.sess.run(gcr._networks['default'].y_mean)
        y_mean_expected = np.zeros_like(y_mean)
        y_std = self.sess.run(gcr._networks['default'].y_std)
        y_std_expected = np.ones_like(y_std)

        assert np.allclose(y_mean, y_mean_expected)
        assert np.allclose(y_std, y_std_expected)

    @pytest.mark.large
    def test_fit_smaller_subsample_factor(self):
        gcr = GaussianCNNBaseline(env_spec=test_env_spec,
                                  filters=((3, (3, 3)), (6, (3, 3))),
                                  strides=(1, 1),
                                  padding='SAME',
                                  hidden_sizes=(32, ),
                                  subsample_factor=0.9,
                                  adaptive_std=False)
        train_data, test_data = get_train_test_data()
        train_paths, _, _ = train_data

        for _ in range(30):
            gcr.fit(train_paths)

        test_paths, expected = test_data

        prediction = gcr.predict(test_paths)
        average_error = 0.0
        for i, exp in enumerate(expected):
            average_error += np.abs(exp - prediction[i])
        average_error /= len(expected)
        assert average_error <= 0.1

    def test_image_input(self):
        env = GymEnv(DummyDiscretePixelEnv(), is_image=True)
        gcb = GaussianCNNBaseline(env_spec=env.spec,
                                  filters=((3, (3, 3)), (6, (3, 3))),
                                  strides=(1, 1),
                                  padding='SAME',
                                  hidden_sizes=(32, ))
        env.reset()
        es = env.step(1)
        obs, rewards = es.observation, es.reward
        train_paths = [{'observations': [obs], 'returns': [rewards]}]
        gcb.fit(train_paths)
        paths = {'observations': [obs]}
        prediction = gcb.predict(paths)
        assert np.allclose(0., prediction)

    @pytest.mark.large
    def test_fit_without_trusted_region(self):
        gcr = GaussianCNNBaseline(env_spec=test_env_spec,
                                  filters=((3, (3, 3)), (6, (3, 3))),
                                  strides=(1, 1),
                                  padding='SAME',
                                  hidden_sizes=(32, ),
                                  adaptive_std=False,
                                  use_trust_region=False)
        train_data, test_data = get_train_test_data()
        train_paths, _, _ = train_data

        for _ in range(20):
            gcr.fit(train_paths)

        test_paths, expected = test_data

        prediction = gcr.predict(test_paths)
        average_error = 0.0
        for i, exp in enumerate(expected):
            average_error += np.abs(exp - prediction[i])
        average_error /= len(expected)
        assert average_error <= 0.1

    @mock.patch('tests.garage.tf.baselines.'
                'test_gaussian_cnn_baseline.'
                'LbfgsOptimizer')
    def test_optimizer_args(self, mock_lbfgs):
        lbfgs_args = dict(max_opt_itr=25)
        gcr = GaussianCNNBaseline(env_spec=test_env_spec,
                                  filters=((3, (3, 3)), (6, (3, 3))),
                                  strides=(1, 1),
                                  padding='SAME',
                                  hidden_sizes=(32, ),
                                  optimizer=LbfgsOptimizer,
                                  optimizer_args=lbfgs_args,
                                  use_trust_region=True)

        assert mock_lbfgs.return_value is gcr._optimizer

        mock_lbfgs.assert_called_with(max_opt_itr=25)

    def test_is_pickleable(self):
        gcr = GaussianCNNBaseline(env_spec=test_env_spec,
                                  filters=((3, (3, 3)), (6, (3, 3))),
                                  strides=(1, 1),
                                  padding='SAME',
                                  hidden_sizes=(32, ),
                                  adaptive_std=False,
                                  use_trust_region=False)

        with tf.compat.v1.variable_scope('GaussianCNNBaseline', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/hidden_0/bias')
        bias.load(tf.ones_like(bias).eval())

        _, test_data = get_train_test_data()
        test_paths, _ = test_data

        result1 = gcr.predict(test_paths)
        h = pickle.dumps(gcr)

        with tf.compat.v1.Session(graph=tf.Graph()):
            gcr_pickled = pickle.loads(h)
            result2 = gcr_pickled.predict(test_paths)
            assert np.array_equal(result1, result2)

    def test_param_values(self):
        gcb = GaussianCNNBaseline(env_spec=test_env_spec,
                                  filters=((3, (3, 3)), (6, (3, 3))),
                                  strides=(1, 1),
                                  padding='SAME',
                                  hidden_sizes=(32, ),
                                  adaptive_std=False,
                                  use_trust_region=False)
        new_gcb = GaussianCNNBaseline(env_spec=test_env_spec,
                                      filters=((3, (3, 3)), (6, (3, 3))),
                                      strides=(1, 1),
                                      padding='SAME',
                                      hidden_sizes=(32, ),
                                      adaptive_std=False,
                                      use_trust_region=False,
                                      name='GaussianCNNBaseline2')

        # Manual change the parameter of GaussianCNNBaseline
        with tf.compat.v1.variable_scope('GaussianCNNBaseline', reuse=True):
            bias_var = tf.compat.v1.get_variable(
                'dist_params/mean_network/hidden_0/bias')
        bias_var.load(tf.ones_like(bias_var).eval())

        old_param_values = gcb.get_param_values()
        new_param_values = new_gcb.get_param_values()
        assert not np.array_equal(old_param_values, new_param_values)
        new_gcb.set_param_values(old_param_values)
        new_param_values = new_gcb.get_param_values()
        assert np.array_equal(old_param_values, new_param_values)

    def test_clone(self):
        gcb = GaussianCNNBaseline(env_spec=test_env_spec,
                                  filters=((3, (3, 3)), (6, (3, 3))),
                                  strides=(1, 1),
                                  padding='SAME',
                                  hidden_sizes=(32, ),
                                  adaptive_std=False,
                                  use_trust_region=False)
        cloned_gcb_model = gcb.clone_model(name='cloned_model')
        for cloned_param, param in zip(cloned_gcb_model.parameters.values(),
                                       gcb.parameters.values()):
            assert np.array_equal(cloned_param, param)
