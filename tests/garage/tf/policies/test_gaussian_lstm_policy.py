import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianLSTMPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestGaussianLSTMPolicy(TfGraphTestCase):

    def test_invalid_env(self):
        env = TfEnv(DummyDiscreteEnv())
        with pytest.raises(ValueError):
            GaussianLSTMPolicy(env_spec=env.spec)

    # yapf: disable
    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim', [
        ((1, ), (1, ), 4),
        ((2, ), (2, ), 4),
        ((1, 1), (1, ), 4),
        ((2, 2), (2, ), 4)
    ])
    # yapf: enable
    def test_get_action_state_include_action(self, obs_dim, action_dim,
                                             hidden_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        obs_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=[
                None, None,
                env.observation_space.flat_dim + np.prod(action_dim)
            ],
            name='obs')
        policy = GaussianLSTMPolicy(env_spec=env.spec,
                                    hidden_dim=hidden_dim,
                                    state_include_action=True)
        policy.build(obs_var)
        policy.reset()
        obs = env.reset()
        action, _ = policy.get_action(obs.flatten())
        assert env.action_space.contains(action)

        policy.reset()

        actions, _ = policy.get_actions([obs.flatten()])
        for action in actions:
            assert env.action_space.contains(action)

    # yapf: disable
    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim', [
        ((1, ), (1, ), 4),
        ((2, ), (2, ), 4),
        ((1, 1), (1, ), 4),
        ((2, 2), (2, ), 4)
    ])
    # yapf: enable
    def test_get_action(self, obs_dim, action_dim, hidden_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        obs_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, None, env.observation_space.flat_dim],
            name='obs')
        policy = GaussianLSTMPolicy(env_spec=env.spec,
                                    hidden_dim=hidden_dim,
                                    state_include_action=False)

        policy.build(obs_var)
        policy.reset()
        obs = env.reset()

        action, _ = policy.get_action(obs.flatten())
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs.flatten()])
        for action in actions:
            assert env.action_space.contains(action)

    def test_is_pickleable(self):
        env = TfEnv(DummyBoxEnv(obs_dim=(1, ), action_dim=(1, )))
        obs_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, None, env.observation_space.flat_dim],
            name='obs')
        policy = GaussianLSTMPolicy(env_spec=env.spec,
                                    state_include_action=False)

        policy.build(obs_var)
        env.reset()
        obs = env.reset()
        with tf.compat.v1.variable_scope(
                'GaussianLSTMPolicy/GaussianLSTMModel', reuse=True):
            param = tf.compat.v1.get_variable(
                'dist_params/log_std_param/parameter')
        # assign it to all one
        param.load(tf.ones_like(param).eval())

        output1 = self.sess.run(
            [policy.distribution.loc,
             policy.distribution.stddev()],
            feed_dict={policy.model.input: [[obs.flatten()], [obs.flatten()]]})

        p = pickle.dumps(policy)
        # yapf: disable
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            obs_var = tf.compat.v1.placeholder(
                        tf.float32,
                        shape=[None, None, env.observation_space.flat_dim],
                        name='obs')
            policy_pickled.build(obs_var)
            output2 = sess.run(
                [
                    policy_pickled.distribution.loc,
                    policy_pickled.distribution.stddev()
                ],
                feed_dict={
                    policy_pickled.model.input: [[obs.flatten()],
                                                 [obs.flatten()]]
                })
            assert np.array_equal(output1, output2)
        # yapf: enable

    def test_state_info_specs(self):
        env = TfEnv(DummyBoxEnv(obs_dim=(4, ), action_dim=(4, )))
        policy = GaussianLSTMPolicy(env_spec=env.spec,
                                    state_include_action=False)
        assert policy.state_info_specs == []

    def test_state_info_specs_with_state_include_action(self):
        env = TfEnv(DummyBoxEnv(obs_dim=(4, ), action_dim=(4, )))
        policy = GaussianLSTMPolicy(env_spec=env.spec,
                                    state_include_action=True)
        assert policy.state_info_specs == [('prev_action', (4, ))]

    def test_clone(self):
        env = TfEnv(DummyBoxEnv(obs_dim=(4, ), action_dim=(4, )))
        policy = GaussianLSTMPolicy(env_spec=env.spec)
        policy_clone = policy.clone('GaussianLSTMPolicyClone')
        assert policy_clone.env_spec == policy.env_spec
