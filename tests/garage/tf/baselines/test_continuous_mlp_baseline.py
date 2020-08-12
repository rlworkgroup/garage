import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.tf.baselines import ContinuousMLPBaseline

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


def get_train_test_data():
    data = np.linspace(-np.pi, np.pi, 1000)
    train_paths = [{
        'observations': [[x - 0.01, x + 0.01]],
        'returns': [np.sin(x)]
    } for x in data]
    observations = np.concatenate([p['observations'] for p in train_paths])

    data_test = [
        -np.pi, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2, np.pi
    ]
    paths = {'observations': [[x - 0.01, x + 0.01] for x in data_test]}
    expected = [0, -1, -0.707, 0, 0.707, 1, 0]

    return train_paths, observations, paths, expected


class TestContinuousMLPBaseline(TfGraphTestCase):

    def test_fit_normalized(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        cmb = ContinuousMLPBaseline(env_spec=box_env_spec)

        train_paths, observations, paths, expected = get_train_test_data()

        for _ in range(20):
            cmb.fit(train_paths)

        prediction = cmb.predict(paths)

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(cmb._x_mean)
        x_mean_expected = np.mean(observations, axis=0, keepdims=True)
        x_std = self.sess.run(cmb._x_std)
        x_std_expected = np.std(observations, axis=0, keepdims=True)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

    def test_fit_unnormalized(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        cmb = ContinuousMLPBaseline(env_spec=box_env_spec,
                                    normalize_inputs=False)
        train_paths, _, paths, expected = get_train_test_data()

        for _ in range(20):
            cmb.fit(train_paths)

        prediction = cmb.predict(paths)

        assert np.allclose(prediction, expected, rtol=0, atol=0.1)

        x_mean = self.sess.run(cmb._x_mean)
        x_mean_expected = np.zeros_like(x_mean)
        x_std = self.sess.run(cmb._x_std)
        x_std_expected = np.ones_like(x_std)

        assert np.allclose(x_mean, x_mean_expected)
        assert np.allclose(x_std, x_std_expected)

    def test_is_pickleable(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        cmb = ContinuousMLPBaseline(env_spec=box_env_spec)

        with tf.compat.v1.variable_scope('ContinuousMLPBaseline', reuse=True):
            bias = tf.compat.v1.get_variable('mlp/hidden_0/bias')
        bias.load(tf.ones_like(bias).eval())

        _, _, paths, _ = get_train_test_data()
        result1 = cmb.predict(paths)
        h = pickle.dumps(cmb)

        with tf.compat.v1.Session(graph=tf.Graph()):
            cmb_pickled = pickle.loads(h)
            result2 = cmb_pickled.predict(paths)
            assert np.array_equal(result1, result2)

    def test_param_values(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        cmb = ContinuousMLPBaseline(env_spec=box_env_spec)
        new_cmb = ContinuousMLPBaseline(env_spec=box_env_spec,
                                        name='ContinuousMLPBaseline2')

        # Manual change the parameter of ContinuousMLPBaseline
        with tf.compat.v1.variable_scope('ContinuousMLPBaseline', reuse=True):
            bias = tf.compat.v1.get_variable('mlp/hidden_0/bias')
        bias.load(tf.ones_like(bias).eval())

        old_param_values = cmb.get_param_values()
        new_param_values = new_cmb.get_param_values()
        assert not np.array_equal(old_param_values, new_param_values)
        new_cmb.set_param_values(old_param_values)
        new_param_values = new_cmb.get_param_values()
        assert np.array_equal(old_param_values, new_param_values)

    def test_get_params(self):
        box_env_spec = GymEnv(DummyBoxEnv(obs_dim=(2, ))).spec
        cmb = ContinuousMLPBaseline(env_spec=box_env_spec)
        params_internal = cmb.get_params()
        trainable_params = tf.compat.v1.trainable_variables(
            scope='ContinuousMLPBaseline')
        assert np.array_equal(params_internal, trainable_params)
