"""
Unit test for ContinuousMLPBaselineWithModel.

This test consists of two ContinuousMLPBaseline's: B1 and B2, and
two ContinuousMLPBaselineWithModel: B3 and B4.

This test ensures the outputs from all the baselines are the same,
for the transition from using ContinuousMLPBaseline to
ContinuousMLPBaselineWithModel.

"""
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.baselines import ContinuousMLPBaseline
from garage.tf.baselines import ContinuousMLPBaselineWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestContinuousMLPPolicyWithModelTransit(TfGraphTestCase):
    def setup_method(self):
        with mock.patch('tensorflow.random.normal') as mock_rand:
            mock_rand.return_value = 0.5
            super().setup_method()
            self.box_env = TfEnv(DummyBoxEnv())
            self.baseline1 = ContinuousMLPBaseline(
                                    self.box_env.spec,
                                    regressor_args=dict(hidden_sizes=(32, 32)),
                                    name="CMB_1")
            self.baseline2 = ContinuousMLPBaseline(
                                    self.box_env.spec,
                                    regressor_args=dict(hidden_sizes=(64, 64)),
                                    name="CMB_2")
            self.baseline3 = ContinuousMLPBaselineWithModel(
                                    self.box_env.spec,
                                    regressor_args=dict(hidden_sizes=(32, 32)),
                                    name="CMB_WM_1")
            self.baseline4 = ContinuousMLPBaselineWithModel(
                                    self.box_env.spec,
                                    regressor_args=dict(hidden_sizes=(64, 64)),
                                    name="CMB_WM_2")

            self.sess.run(tf.compat.v1.global_variables_initializer())
            for a, b in zip(self.baseline1.get_params_internal(),
                            self.baseline3.get_params_internal()):
                self.sess.run(a.assign(b))

            for a, b in zip(self.baseline4.get_params_internal(),
                            self.baseline2.get_params_internal()):
                self.sess.run(a.assign(b))
            assert(np.array_equal(self.baseline4.get_param_values(), self.baseline2.get_param_values()))
            assert(np.array_equal(self.baseline1.get_param_values(), self.baseline3.get_param_values()))

    def test_transition(self):
        """ Test to see that CMLPBase with and without model have the same outputs.
        Fit to the same mock observations.
        Check that there is same output.
        """
        obs_dim = self.box_env.observation_space.shape
        # with mock.patch('numpy.random.normal') as mock_rand:
        #     mock_rand.return_value = 0.5
        paths = [{'observations': [np.full(shape=obs_dim, fill_value=1)],
                    'returns': [1]}, 
                    {'observations': [np.full(shape=obs_dim, fill_value=2)],
                    'returns': [2]}
                ]
        
        obs = {'observations': [np.full(obs_dim, 1), np.full(obs_dim, 2)]}
        prediction1 = self.baseline1.predict(obs)
        prediction2 = self.baseline2.predict(obs)
        prediction3 = self.baseline3.predict(obs)
        prediction4 = self.baseline4.predict(obs)

        # check before fitting that predictions are the same
        # between model vs not model

        # assert np.allclose(prediction1, prediction3, rtol=1e-6)
        # assert np.allclose(prediction2, prediction4, rtol=1e-6)
        import ipdb; ipdb.set_trace()

        self.baseline1.fit(paths)
        self.baseline2.fit(paths)
        self.baseline3.fit(paths)
        self.baseline4.fit(paths)

        obs = {'observations': [np.full(obs_dim, 1), np.full(obs_dim, 2)]}
        prediction1 = self.baseline1.predict(obs)
        prediction2 = self.baseline2.predict(obs)
        prediction3 = self.baseline3.predict(obs)
        prediction4 = self.baseline4.predict(obs)

        assert np.array_equal(prediction1, prediction3)
        assert np.array_equal(prediction2, prediction4)


    # def test_get_action_sym(self):
    #     obs_dim = self.box_env.spec.observation_space.flat_dim
    #     state_input = tf.compat.v1.placeholder(
    #         tf.float32, shape=(None, obs_dim))

    #     action_sym1 = self.policy1.get_action_sym(
    #         state_input, name='action_sym')
    #     action_sym2 = self.policy2.get_action_sym(
    #         state_input, name='action_sym')
    #     action_sym3 = self.policy3.get_action_sym(
    #         state_input, name='action_sym')
    #     action_sym4 = self.policy4.get_action_sym(
    #         state_input, name='action_sym')

    #     action1 = self.sess.run(
    #         action_sym1, feed_dict={state_input: [self.obs]})
    #     action2 = self.sess.run(
    #         action_sym2, feed_dict={state_input: [self.obs]})
    #     action3 = self.sess.run(
    #         action_sym3, feed_dict={state_input: [self.obs]})
    #     action4 = self.sess.run(
    #         action_sym4, feed_dict={state_input: [self.obs]})

    #     assert np.array_equal(action1, action3 * self.action_bound)
    #     assert np.array_equal(action2, action4 * self.action_bound)