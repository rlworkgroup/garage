import pickle
from unittest import mock

from nose2.tools.params import params
import numpy as np
import tensorflow as tf

from garage.tf.baselines import DeterministicMLPBaselineWithModel
from garage.tf.envs import TfEnv
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.regressors import SimpleMLPRegressor


class TestDeterministicMLPBaselineWithModel(TfGraphTestCase):
    @params([1], [2], [1, 1], [2, 2])
    def test_fit(self, obs_dim):
        box_env = TfEnv(DummyBoxEnv(obs_dim=obs_dim))
        with mock.patch(('garage.tf.baselines.'
                         'deterministic_mlp_baseline_with_model.'
                         'DeterministicMLPRegressorWithModel'),
                        new=SimpleMLPRegressor):
            gmb = DeterministicMLPBaselineWithModel(env_spec=box_env.spec)
        paths = [{
            'observations': [np.full(obs_dim, 1)],
            'returns': [1]
        }, {
            'observations': [np.full(obs_dim, 2)],
            'returns': [2]
        }]
        gmb.fit(paths)

        obs = {'observations': [np.full(obs_dim, 1), np.full(obs_dim, 2)]}
        prediction = gmb.predict(obs)
        assert np.array_equal(prediction, [1, 2])

    @params([1], [2], [1, 1], [2, 2])
    def test_param_values(self, obs_dim):
        box_env = TfEnv(DummyBoxEnv(obs_dim=obs_dim))
        with mock.patch(('garage.tf.baselines.'
                         'deterministic_mlp_baseline_with_model.'
                         'DeterministicMLPRegressorWithModel'),
                        new=SimpleMLPRegressor):
            gmb = DeterministicMLPBaselineWithModel(env_spec=box_env.spec)
            new_gmb = DeterministicMLPBaselineWithModel(
                env_spec=box_env.spec,
                name='DeterministicMLPBaselineWithModel2')

        # Manual change the parameter of DeterministicMLPBaselineWithModel
        with tf.variable_scope(
                'DeterministicMLPBaselineWithModel2', reuse=True):
            return_var = tf.get_variable('SimpleMLPModel/return_var')
        return_var.load(1.0)

        old_param_values = gmb.get_param_values()
        new_param_values = new_gmb.get_param_values()
        assert not np.array_equal(old_param_values, new_param_values)
        new_gmb.set_param_values(old_param_values)
        new_param_values = new_gmb.get_param_values()
        assert np.array_equal(old_param_values, new_param_values)

    @params([1], [2], [1, 1], [2, 2])
    def test_get_params_internal(self, obs_dim):
        box_env = TfEnv(DummyBoxEnv(obs_dim=obs_dim))
        with mock.patch(('garage.tf.baselines.'
                         'deterministic_mlp_baseline_with_model.'
                         'DeterministicMLPRegressorWithModel'),
                        new=SimpleMLPRegressor):
            gmb = DeterministicMLPBaselineWithModel(env_spec=box_env.spec)
        params_interal = gmb.get_params_internal()
        trainable_params = tf.trainable_variables(
            scope='DeterministicMLPBaselineWithModel')
        assert np.array_equal(params_interal, trainable_params)

    def test_is_pickleable(self):
        box_env = TfEnv(DummyBoxEnv(obs_dim=(1, )))
        with mock.patch(('garage.tf.baselines.'
                         'deterministic_mlp_baseline_with_model.'
                         'DeterministicMLPRegressorWithModel'),
                        new=SimpleMLPRegressor):
            gmb = DeterministicMLPBaselineWithModel(env_spec=box_env.spec)
        obs = {'observations': [np.full(1, 1), np.full(1, 1)]}

        with tf.variable_scope(
                'DeterministicMLPBaselineWithModel', reuse=True):
            return_var = tf.get_variable('SimpleMLPModel/return_var')
        return_var.load(1.0)

        prediction = gmb.predict(obs)

        h = pickle.dumps(gmb)

        with tf.Session(graph=tf.Graph()):
            gmb_pickled = pickle.loads(h)
            prediction2 = gmb_pickled.predict(obs)

            assert np.array_equal(prediction, prediction2)
