import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.tf.baselines import GaussianMLPBaseline

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


def get_train_test_data():
    data = np.linspace(-np.pi, np.pi, 1000)
    train_paths = [{
        'observations': [[x - 0.01, x + 0.01]],
        'returns': [np.sin(x)]
    } for x in data]
    observations = np.concatenate([p['observations'] for p in train_paths])
    returns = np.concatenate([p['returns'] for p in train_paths])
    returns = returns.reshape((-1, 1))

    data_test = [
        -np.pi, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, np.pi
    ]
    paths = {'observations': [[x - 0.01, x + 0.01] for x in data_test]}
    expected = [0, -1, -0.707, 0, 0.707, 1, 0]

    return train_paths, observations, returns, paths, expected


class TestGaussianMLPBaseline(TfGraphTestCase):

    # unmarked to balance test jobs
    # @pytest.mark.large
    def test_fit_normalized(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        gmb = GaussianMLPBaseline(env_spec=box_env_spec)

        (train_paths, observations, returns, paths,
         expected) = get_train_test_data()

        for _ in range(150):
            gmb.fit(train_paths)

        prediction = gmb.predict(paths)

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(gmb._networks['default'].x_mean)
        x_mean_expected = np.mean(observations, axis=0, keepdims=True)
        x_std = self.sess.run(gmb._networks['default'].x_std)
        x_std_expected = np.std(observations, axis=0, keepdims=True)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

        y_mean = self.sess.run(gmb._networks['default'].y_mean)
        y_mean_expected = np.mean(returns, axis=0, keepdims=True)
        y_std = self.sess.run(gmb._networks['default'].y_std)
        y_std_expected = np.std(returns, axis=0, keepdims=True)

        assert np.allclose(y_mean, y_mean_expected)
        assert np.allclose(y_std, y_std_expected)

    # unmarked to balance test jobs
    # @pytest.mark.large
    def test_fit_unnormalized(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        gmb = GaussianMLPBaseline(env_spec=box_env_spec,
                                  subsample_factor=0.9,
                                  normalize_inputs=False,
                                  normalize_outputs=False)

        train_paths, _, _, paths, expected = get_train_test_data()

        for _ in range(150):
            gmb.fit(train_paths)

        prediction = gmb.predict(paths)

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(gmb._networks['default'].x_mean)
        x_mean_expected = np.zeros_like(x_mean)
        x_std = self.sess.run(gmb._networks['default'].x_std)
        x_std_expected = np.ones_like(x_std)
        assert np.array_equal(x_mean, x_mean_expected)
        assert np.array_equal(x_std, x_std_expected)

        y_mean = self.sess.run(gmb._networks['default'].y_mean)
        y_mean_expected = np.zeros_like(y_mean)
        y_std = self.sess.run(gmb._networks['default'].y_std)
        y_std_expected = np.ones_like(y_std)

        assert np.allclose(y_mean, y_mean_expected)
        assert np.allclose(y_std, y_std_expected)

    # unmarked to balance test jobs
    # @pytest.mark.large
    def test_fit_smaller_subsample_factor(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        gmb = GaussianMLPBaseline(env_spec=box_env_spec, subsample_factor=0.9)

        train_paths, _, _, paths, expected = get_train_test_data()

        for _ in range(150):
            gmb.fit(train_paths)

        prediction = gmb.predict(paths)

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

    # unmarked to balance test jobs
    # @pytest.mark.large
    def test_fit_without_trusted_region(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        gmb = GaussianMLPBaseline(env_spec=box_env_spec,
                                  use_trust_region=False)

        train_paths, _, _, paths, expected = get_train_test_data()

        for _ in range(150):
            gmb.fit(train_paths)

        prediction = gmb.predict(paths)

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

    def test_param_values(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        gmb = GaussianMLPBaseline(env_spec=box_env_spec)
        new_gmb = GaussianMLPBaseline(env_spec=box_env_spec,
                                      name='GaussianMLPBaseline2')

        # Manual change the parameter of GaussianMLPBaseline
        with tf.compat.v1.variable_scope('GaussianMLPBaseline', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/hidden_0/bias')
        bias.load(tf.ones_like(bias).eval())

        old_param_values = gmb.get_param_values()
        new_param_values = new_gmb.get_param_values()
        assert not np.array_equal(old_param_values, new_param_values)
        new_gmb.set_param_values(old_param_values)
        new_param_values = new_gmb.get_param_values()
        assert np.array_equal(old_param_values, new_param_values)

    def test_is_pickleable(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        gmb = GaussianMLPBaseline(env_spec=box_env_spec)
        _, _, _, paths, _ = get_train_test_data()

        with tf.compat.v1.variable_scope('GaussianMLPBaseline', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/hidden_0/bias')
        bias.load(tf.ones_like(bias).eval())

        prediction = gmb.predict(paths)

        h = pickle.dumps(gmb)

        with tf.compat.v1.Session(graph=tf.Graph()):
            gmb_pickled = pickle.loads(h)
            prediction2 = gmb_pickled.predict(paths)

            assert np.array_equal(prediction, prediction2)

    def test_clone(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        gmb = GaussianMLPBaseline(env_spec=box_env_spec)
        cloned_gmb_model = gmb.clone_model(name='cloned_model')
        for cloned_param, param in zip(cloned_gmb_model.parameters.values(),
                                       gmb.parameters.values()):
            assert np.array_equal(cloned_param, param)
