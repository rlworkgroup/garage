import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.baselines import ContinuousMLPBaseline
from garage.tf.envs import TfEnv
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.regressors import SimpleMLPRegressor


class TestContinuousMLPBaseline(TfGraphTestCase):

    @pytest.mark.parametrize('obs_dim', [[1], [2], [1, 1], [2, 2]])
    def test_fit(self, obs_dim):
        box_env = TfEnv(DummyBoxEnv(obs_dim=obs_dim))
        with mock.patch(('garage.tf.baselines.'
                         'continuous_mlp_baseline.'
                         'ContinuousMLPRegressor'),
                        new=SimpleMLPRegressor):
            cmb = ContinuousMLPBaseline(env_spec=box_env.spec)
        paths = [{
            'observations': [np.full(obs_dim, 1)],
            'returns': [1]
        }, {
            'observations': [np.full(obs_dim, 2)],
            'returns': [2]
        }]
        cmb.fit(paths)

        obs = {'observations': [np.full(obs_dim, 1), np.full(obs_dim, 2)]}
        prediction = cmb.predict(obs)
        assert np.array_equal(prediction, [1, 2])

    @pytest.mark.parametrize('obs_dim', [[1], [2], [1, 1], [2, 2]])
    def test_param_values(self, obs_dim):
        box_env = TfEnv(DummyBoxEnv(obs_dim=obs_dim))
        with mock.patch(('garage.tf.baselines.'
                         'continuous_mlp_baseline.'
                         'ContinuousMLPRegressor'),
                        new=SimpleMLPRegressor):
            cmb = ContinuousMLPBaseline(env_spec=box_env.spec)
            new_cmb = ContinuousMLPBaseline(env_spec=box_env.spec,
                                            name='ContinuousMLPBaseline2')

        # Manual change the parameter of ContinuousMLPBaseline
        with tf.compat.v1.variable_scope('ContinuousMLPBaseline2', reuse=True):
            return_var = tf.compat.v1.get_variable('SimpleMLPModel/return_var')
        return_var.load(1.0)

        old_param_values = cmb.get_param_values()
        new_param_values = new_cmb.get_param_values()
        assert not np.array_equal(old_param_values, new_param_values)
        new_cmb.set_param_values(old_param_values)
        new_param_values = new_cmb.get_param_values()
        assert np.array_equal(old_param_values, new_param_values)

    @pytest.mark.parametrize('obs_dim', [[1], [2], [1, 1], [2, 2]])
    def test_get_params_internal(self, obs_dim):
        box_env = TfEnv(DummyBoxEnv(obs_dim=obs_dim))
        with mock.patch(('garage.tf.baselines.'
                         'continuous_mlp_baseline.'
                         'ContinuousMLPRegressor'),
                        new=SimpleMLPRegressor):
            cmb = ContinuousMLPBaseline(env_spec=box_env.spec)
        params_interal = cmb.get_params_internal()
        trainable_params = tf.compat.v1.trainable_variables(
            scope='ContinuousMLPBaseline')
        assert np.array_equal(params_interal, trainable_params)

    def test_is_pickleable(self):
        box_env = TfEnv(DummyBoxEnv(obs_dim=(1, )))
        with mock.patch(('garage.tf.baselines.'
                         'continuous_mlp_baseline.'
                         'ContinuousMLPRegressor'),
                        new=SimpleMLPRegressor):
            cmb = ContinuousMLPBaseline(env_spec=box_env.spec)
        obs = {'observations': [np.full(1, 1), np.full(1, 1)]}

        with tf.compat.v1.variable_scope('ContinuousMLPBaseline', reuse=True):
            return_var = tf.compat.v1.get_variable('SimpleMLPModel/return_var')
        return_var.load(1.0)

        prediction = cmb.predict(obs)

        h = pickle.dumps(cmb)

        with tf.compat.v1.Session(graph=tf.Graph()):
            cmb_pickled = pickle.loads(h)
            prediction2 = cmb_pickled.predict(obs)

            assert np.array_equal(prediction, prediction2)
