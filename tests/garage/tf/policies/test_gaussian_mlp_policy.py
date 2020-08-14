import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.tf.policies import GaussianMLPPolicy

# yapf: disable
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import (DummyBoxEnv,
                                       DummyDictEnv,
                                       DummyDiscreteEnv)

# yapf: enable


class TestGaussianMLPPolicy(TfGraphTestCase):

    def test_invalid_env(self):
        env = GymEnv(DummyDiscreteEnv())
        with pytest.raises(ValueError):
            GaussianMLPPolicy(env_spec=env.spec)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_get_action(self, obs_dim, action_dim):
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = GaussianMLPPolicy(env_spec=env.spec)

        env.reset()
        obs = env.step(1).observation

        action, _ = policy.get_action(obs.flatten())
        assert env.action_space.contains(action)
        actions, _ = policy.get_actions(
            [obs.flatten(), obs.flatten(),
             obs.flatten()])
        for action in actions:
            assert env.action_space.contains(action)

    def test_get_action_dict_space(self):
        env = GymEnv(DummyDictEnv(obs_space_type='box', act_space_type='box'))
        policy = GaussianMLPPolicy(env_spec=env.spec)
        obs = env.reset()[0]

        action, _ = policy.get_action(obs)
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs, obs])
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
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = GaussianMLPPolicy(env_spec=env.spec)
        obs = env.reset()[0]

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist
        dist_sym2 = policy.build(state_input, name='dist_sym2').dist
        output1 = self.sess.run([dist_sym.loc],
                                feed_dict={state_input: [[obs.flatten()]]})
        output2 = self.sess.run([dist_sym2.loc],
                                feed_dict={state_input: [[obs.flatten()]]})
        assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_is_pickleable(self, obs_dim, action_dim):
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = GaussianMLPPolicy(env_spec=env.spec)

        obs = env.reset()[0]

        with tf.compat.v1.variable_scope('GaussianMLPPolicy', reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/hidden_0/bias')
        # assign it to all one
        bias.load(tf.ones_like(bias).eval())

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist
        output1 = self.sess.run([dist_sym.loc, dist_sym.stddev()],
                                feed_dict={state_input: [[obs.flatten()]]})

        p = pickle.dumps(policy)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            state_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, None,
                                                          policy.input_dim))
            dist_sym = policy_pickled.build(state_input, name='dist_sym').dist
            output2 = sess.run([dist_sym.loc, dist_sym.stddev()],
                               feed_dict={state_input: [[obs.flatten()]]})
            assert np.array_equal(output1, output2)

    def test_clone(self):
        env = GymEnv(DummyBoxEnv(obs_dim=(10, ), action_dim=(4, )))
        policy = GaussianMLPPolicy(env_spec=env.spec)
        policy_clone = policy.clone('GaussnaMLPPolicyClone')
        assert policy.env_spec == policy_clone.env_spec
        for cloned_param, param in zip(policy_clone.parameters.values(),
                                       policy.parameters.values()):
            assert np.array_equal(cloned_param, param)
