import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GarageEnv
from garage.tf.policies import CategoricalMLPPolicy

# yapf: disable
from tests.fixtures import TfGraphTestCase  # noqa:I202
from tests.fixtures.envs.dummy import (DummyBoxEnv,
                                       DummyDictEnv,
                                       DummyDiscreteEnv)

# yapf: enable


class TestCategoricalMLPPolicy(TfGraphTestCase):

    def test_invalid_env(self):
        env = GarageEnv(DummyBoxEnv())
        with pytest.raises(ValueError):
            CategoricalMLPPolicy(env_spec=env.spec)

    @pytest.mark.parametrize('obs_dim, action_dim, obs_type', [
        ((1, ), 1, 'discrete'),
        ((2, ), 2, 'discrete'),
        ((1, 1), 1, 'discrete'),
        ((2, 2), 2, 'discrete'),
        ((1, ), 1, 'dict'),
    ])
    def test_get_action(self, obs_dim, action_dim, obs_type):
        assert obs_type in ['discrete', 'dict']
        if obs_type == 'discrete':
            env = GarageEnv(
                DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        else:
            env = GarageEnv(
                DummyDictEnv(obs_space_type='box', act_space_type='discrete'))
        policy = CategoricalMLPPolicy(env_spec=env.spec)
        obs = env.reset()
        if obs_type == 'discrete':
            obs = obs.flatten()
        action, _ = policy.get_action(obs)
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs, obs, obs])
        for action in actions:
            assert env.action_space.contains(action)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_build(self, obs_dim, action_dim):
        env = GarageEnv(
            DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalMLPPolicy(env_spec=env.spec)
        obs = env.reset()

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist
        output1 = self.sess.run(
            [policy.distribution.probs],
            feed_dict={policy.model.input: [[obs.flatten()]]})
        output2 = self.sess.run([dist_sym.probs],
                                feed_dict={state_input: [[obs.flatten()]]})
        assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_is_pickleable(self, obs_dim, action_dim):
        env = GarageEnv(
            DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalMLPPolicy(env_spec=env.spec)
        obs = env.reset()

        with tf.compat.v1.variable_scope(
                'CategoricalMLPPolicy/CategoricalMLPModel', reuse=True):
            bias = tf.compat.v1.get_variable('mlp/hidden_0/bias')
        # assign it to all one
        bias.load(tf.ones_like(bias).eval())
        output1 = self.sess.run(
            [policy.distribution.probs],
            feed_dict={policy.model.input: [[obs.flatten()]]})

        p = pickle.dumps(policy)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            output2 = sess.run(
                [policy_pickled.distribution.probs],
                feed_dict={policy_pickled.model.input: [[obs.flatten()]]})
            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_get_regularizable_vars(self, obs_dim, action_dim):
        env = GarageEnv(
            DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalMLPPolicy(env_spec=env.spec)
        reg_vars = policy.get_regularizable_vars()
        assert len(reg_vars) == 2
        for var in reg_vars:
            assert ('bias' not in var.name) and ('output' not in var.name)
