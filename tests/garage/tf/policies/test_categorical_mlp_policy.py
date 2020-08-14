import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.tf.policies import CategoricalMLPPolicy

# yapf: disable
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import (DummyBoxEnv,
                                       DummyDictEnv,
                                       DummyDiscreteEnv)

# yapf: enable


class TestCategoricalMLPPolicy(TfGraphTestCase):

    def test_invalid_env(self):
        env = GymEnv(DummyBoxEnv())
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
            env = GymEnv(
                DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        else:
            env = GymEnv(
                DummyDictEnv(obs_space_type='box', act_space_type='discrete'))
        policy = CategoricalMLPPolicy(env_spec=env.spec)
        obs = env.reset()[0]
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
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalMLPPolicy(env_spec=env.spec)
        obs = env.reset()[0]

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist
        dist_sym2 = policy.build(state_input, name='dist_sym2').dist
        output1 = self.sess.run([dist_sym.probs],
                                feed_dict={state_input: [[obs.flatten()]]})
        output2 = self.sess.run([dist_sym2.probs],
                                feed_dict={state_input: [[obs.flatten()]]})
        assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_is_pickleable(self, obs_dim, action_dim):
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalMLPPolicy(env_spec=env.spec)
        obs = env.reset()[0]

        with tf.compat.v1.variable_scope('CategoricalMLPPolicy', reuse=True):
            bias = tf.compat.v1.get_variable('mlp/hidden_0/bias')
        # assign it to all one
        bias.load(tf.ones_like(bias).eval())
        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist
        output1 = self.sess.run([dist_sym.probs],
                                feed_dict={state_input: [[obs.flatten()]]})

        p = pickle.dumps(policy)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            state_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, None,
                                                          policy.input_dim))
            dist_sym = policy_pickled.build(state_input, name='dist_sym').dist
            output2 = sess.run([dist_sym.probs],
                               feed_dict={state_input: [[obs.flatten()]]})
            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_get_regularizable_vars(self, obs_dim, action_dim):
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalMLPPolicy(env_spec=env.spec)
        reg_vars = policy.get_regularizable_vars()
        assert len(reg_vars) == 2
        for var in reg_vars:
            assert ('bias' not in var.name) and ('output' not in var.name)

    def test_clone(self):
        env = GymEnv(DummyDiscreteEnv(obs_dim=(10, ), action_dim=4))
        policy = CategoricalMLPPolicy(env_spec=env.spec)
        policy_clone = policy.clone('CategoricalMLPPolicyClone')
        assert policy.env_spec == policy_clone.env_spec
        for cloned_param, param in zip(policy_clone.parameters.values(),
                                       policy.parameters.values()):
            assert np.array_equal(cloned_param, param)
