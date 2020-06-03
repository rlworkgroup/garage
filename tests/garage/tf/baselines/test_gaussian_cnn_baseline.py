import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GarageEnv
from garage.misc.tensor_utils import normalize_pixel_batch
from garage.tf.baselines import GaussianCNNBaseline
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv
from tests.fixtures.regressors import SimpleGaussianCNNRegressor


class TestGaussianCNNBaseline(TfGraphTestCase):

    @pytest.mark.parametrize('obs_dim', [[1, 1, 1], [2, 2, 2], [1, 1], [2, 2]])
    def test_fit(self, obs_dim):
        box_env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim))
        with mock.patch(('garage.tf.baselines.'
                         'gaussian_cnn_baseline.'
                         'GaussianCNNRegressor'),
                        new=SimpleGaussianCNNRegressor):
            gcb = GaussianCNNBaseline(env_spec=box_env.spec)
        paths = [{
            'observations': [np.full(obs_dim, 1)],
            'returns': [1]
        }, {
            'observations': [np.full(obs_dim, 2)],
            'returns': [2]
        }]
        gcb.fit(paths)

        obs = {'observations': [np.full(obs_dim, 1), np.full(obs_dim, 2)]}
        prediction = gcb.predict(obs)
        assert np.array_equal(prediction, [1, 2])

    @pytest.mark.parametrize('obs_dim', [[1], [2], [1, 1, 1, 1], [2, 2, 2, 2]])
    def test_invalid_obs_shape(self, obs_dim):
        box_env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim))
        with pytest.raises(ValueError):
            GaussianCNNBaseline(env_spec=box_env.spec)

    def test_obs_is_image(self):
        env = GarageEnv(DummyDiscretePixelEnv(), is_image=True)
        with mock.patch(('garage.tf.baselines.'
                         'gaussian_cnn_baseline.'
                         'GaussianCNNRegressor'),
                        new=SimpleGaussianCNNRegressor):
            with mock.patch(
                    'garage.tf.baselines.'
                    'gaussian_cnn_baseline.'
                    'normalize_pixel_batch',
                    side_effect=normalize_pixel_batch) as npb:

                gcb = GaussianCNNBaseline(env_spec=env.spec)

                obs_dim = env.spec.observation_space.shape
                paths = [{
                    'observations': [np.full(obs_dim, 1)],
                    'returns': [1]
                }, {
                    'observations': [np.full(obs_dim, 2)],
                    'returns': [2]
                }]

                gcb.fit(paths)
                observations = np.concatenate(
                    [p['observations'] for p in paths])
                assert npb.call_count == 1, (
                    "Expected '%s' to have been called once. Called %s times."
                    % (npb._mock_name or 'mock', npb.call_count))
                assert (npb.call_args_list[0][0][0] == observations).all()

                obs = {
                    'observations': [np.full(obs_dim, 1),
                                     np.full(obs_dim, 2)]
                }
                observations = obs['observations']
                gcb.predict(obs)
                assert npb.call_args_list[1][0][0] == observations

    def test_obs_not_image(self):
        env = GarageEnv(DummyDiscretePixelEnv(), is_image=False)
        with mock.patch(('garage.tf.baselines.'
                         'gaussian_cnn_baseline.'
                         'GaussianCNNRegressor'),
                        new=SimpleGaussianCNNRegressor):
            with mock.patch(
                    'garage.tf.baselines.'
                    'gaussian_cnn_baseline.'
                    'normalize_pixel_batch',
                    side_effect=normalize_pixel_batch) as npb:

                gcb = GaussianCNNBaseline(env_spec=env.spec)

                obs_dim = env.spec.observation_space.shape
                paths = [{
                    'observations': [np.full(obs_dim, 1)],
                    'returns': [1]
                }, {
                    'observations': [np.full(obs_dim, 2)],
                    'returns': [2]
                }]

                gcb.fit(paths)
                obs = {
                    'observations': [np.full(obs_dim, 1),
                                     np.full(obs_dim, 2)]
                }
                gcb.predict(obs)
                assert not npb.called

    @pytest.mark.parametrize('obs_dim', [[1, 1, 1], [2, 2, 2], [1, 1], [2, 2]])
    def test_param_values(self, obs_dim):
        box_env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim))
        with mock.patch(('garage.tf.baselines.'
                         'gaussian_cnn_baseline.'
                         'GaussianCNNRegressor'),
                        new=SimpleGaussianCNNRegressor):
            gcb = GaussianCNNBaseline(env_spec=box_env.spec)
            new_gcb = GaussianCNNBaseline(env_spec=box_env.spec,
                                          name='GaussianCNNBaseline2')

        # Manual change the parameter of GaussianCNNBaseline
        with tf.compat.v1.variable_scope('GaussianCNNBaseline', reuse=True):
            return_var = tf.compat.v1.get_variable(
                'SimpleGaussianCNNModel/return_var')
        return_var.load(1.0)

        old_param_values = gcb.get_param_values()
        new_param_values = new_gcb.get_param_values()
        assert not np.array_equal(old_param_values, new_param_values)
        new_gcb.set_param_values(old_param_values)
        new_param_values = new_gcb.get_param_values()
        assert np.array_equal(old_param_values, new_param_values)

    @pytest.mark.parametrize('obs_dim', [[1, 1, 1], [2, 2, 2], [1, 1], [2, 2]])
    def test_get_params_internal(self, obs_dim):
        box_env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim))
        with mock.patch(('garage.tf.baselines.'
                         'gaussian_cnn_baseline.'
                         'GaussianCNNRegressor'),
                        new=SimpleGaussianCNNRegressor):
            gcb = GaussianCNNBaseline(env_spec=box_env.spec,
                                      regressor_args=dict())
        params_interal = gcb.get_params_internal()
        trainable_params = tf.compat.v1.trainable_variables(
            scope='GaussianCNNBaseline')
        assert np.array_equal(params_interal, trainable_params)

    def test_is_pickleable(self):
        box_env = GarageEnv(DummyBoxEnv(obs_dim=(1, 1)))
        with mock.patch(('garage.tf.baselines.'
                         'gaussian_cnn_baseline.'
                         'GaussianCNNRegressor'),
                        new=SimpleGaussianCNNRegressor):
            gcb = GaussianCNNBaseline(env_spec=box_env.spec)
        obs = {'observations': [np.full((1, 1), 1), np.full((1, 1), 1)]}

        with tf.compat.v1.variable_scope('GaussianCNNBaseline', reuse=True):
            return_var = tf.compat.v1.get_variable(
                'SimpleGaussianCNNModel/return_var')
        return_var.load(1.0)

        prediction = gcb.predict(obs)

        h = pickle.dumps(gcb)

        with tf.compat.v1.Session(graph=tf.Graph()):
            gcb_pickled = pickle.loads(h)
            prediction2 = gcb_pickled.predict(obs)

            assert np.array_equal(prediction, prediction2)
