"""Tests for garage.tf.policies.ContinuousMLPPolicy"""
import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GarageEnv
from garage.tf.policies import ContinuousMLPPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestContinuousMLPPolicy(TfGraphTestCase):
    """Test class for ContinuousMLPPolicy"""

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_get_action(self, obs_dim, action_dim):
        """Test get_action method"""
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = ContinuousMLPPolicy(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        action, _ = policy.get_action(obs.flatten())

        assert env.action_space.contains(action)

        actions, _ = policy.get_actions(
            [obs.flatten(), obs.flatten(),
             obs.flatten()])
        for action in actions:
            assert env.action_space.contains(action)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_build(self, obs_dim, action_dim):
        """Test build method"""
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = ContinuousMLPPolicy(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        obs_dim = env.spec.observation_space.flat_dim
        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, obs_dim))
        action_sym = policy.build(state_input, name='action_sym')

        action = self.sess.run(action_sym,
                               feed_dict={state_input: [obs.flatten()]})
        action = policy.action_space.unflatten(action)

        assert env.action_space.contains(action)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_is_pickleable(self, obs_dim, action_dim):
        """Test if ContinuousMLPPolicy is pickleable"""
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = ContinuousMLPPolicy(env_spec=env.spec)
        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, np.prod(obs_dim)))
        outputs = policy.build(state_input, name='policy')
        env.reset()
        obs, _, _, _ = env.step(1)

        with tf.compat.v1.variable_scope('ContinuousMLPPolicy', reuse=True):
            bias = tf.compat.v1.get_variable('mlp/hidden_0/bias')
        # assign it to all one
        bias.load(tf.ones_like(bias).eval())
        output1 = self.sess.run([outputs],
                                feed_dict={state_input: [obs.flatten()]})

        p = pickle.dumps(policy)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            state_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None,
                                                          np.prod(obs_dim)))
            outputs = policy_pickled.build(state_input, name='policy')
            output2 = sess.run([outputs],
                               feed_dict={state_input: [obs.flatten()]})
            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
    ])
    def test_get_regularizable_vars(self, obs_dim, action_dim):
        """Test get_regularizable_vars method"""
        env = GarageEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = ContinuousMLPPolicy(env_spec=env.spec)
        reg_vars = policy.get_regularizable_vars()
        assert len(reg_vars) == 2
        for var in reg_vars:
            assert ('bias' not in var.name) and ('output' not in var.name)
