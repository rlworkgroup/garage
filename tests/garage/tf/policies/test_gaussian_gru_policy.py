import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.tf.policies import GaussianGRUPolicy

# yapf: disable
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import (DummyBoxEnv,
                                       DummyDictEnv,
                                       DummyDiscreteEnv)

# yapf: enable


class TestGaussianGRUPolicy(TfGraphTestCase):

    def test_invalid_env(self):
        env = GymEnv(DummyDiscreteEnv())
        with pytest.raises(ValueError):
            GaussianGRUPolicy(env_spec=env.spec)

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
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = GaussianGRUPolicy(env_spec=env.spec,
                                   hidden_dim=hidden_dim,
                                   state_include_action=True)
        policy.reset()
        obs = env.reset()[0]

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
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = GaussianGRUPolicy(env_spec=env.spec,
                                   hidden_dim=hidden_dim,
                                   state_include_action=False)
        policy.reset(do_resets=None)
        obs = env.reset()[0]

        action, _ = policy.get_action(obs.flatten())
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs.flatten()])
        for action in actions:
            assert env.action_space.contains(action)

    def test_get_action_dict_space(self):
        env = GymEnv(DummyDictEnv(obs_space_type='box', act_space_type='box'))
        policy = GaussianGRUPolicy(env_spec=env.spec,
                                   hidden_dim=4,
                                   state_include_action=False)
        policy.reset(do_resets=None)
        obs = env.reset()[0]

        action, _ = policy.get_action(obs)
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs, obs])
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
    # pylint: disable=no-member
    def test_build_state_include_action(self, obs_dim, action_dim, hidden_dim):
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = GaussianGRUPolicy(env_spec=env.spec,
                                   hidden_dim=hidden_dim,
                                   state_include_action=True)
        policy.reset(do_resets=None)
        obs = env.reset()[0]

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist
        dist_sym2 = policy.build(state_input, name='dist_sym2').dist

        concat_obs = np.concatenate([obs.flatten(), np.zeros(action_dim)])
        output1 = self.sess.run(
            [dist_sym.loc],
            feed_dict={state_input: [[concat_obs], [concat_obs]]})
        output2 = self.sess.run(
            [dist_sym2.loc],
            feed_dict={state_input: [[concat_obs], [concat_obs]]})
        assert np.array_equal(output1, output2)

    # yapf: disable
    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim', [
        ((1, ), (1, ), 4),
        ((2, ), (2, ), 4),
        ((1, 1), (1, ), 4),
        ((2, 2), (2, ), 4)
    ])
    # yapf: enable
    # pylint: disable=no-member
    def test_build_state_not_include_action(self, obs_dim, action_dim,
                                            hidden_dim):
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = GaussianGRUPolicy(env_spec=env.spec,
                                   hidden_dim=hidden_dim,
                                   state_include_action=False)
        policy.reset(do_resets=None)
        obs = env.reset()[0]

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist
        dist_sym2 = policy.build(state_input, name='dist_sym2').dist

        output1 = self.sess.run(
            [dist_sym.loc],
            feed_dict={state_input: [[obs.flatten()], [obs.flatten()]]})
        output2 = self.sess.run(
            [dist_sym2.loc],
            feed_dict={state_input: [[obs.flatten()], [obs.flatten()]]})
        assert np.array_equal(output1, output2)

    # pylint: disable=no-member
    def test_is_pickleable(self):
        env = GymEnv(DummyBoxEnv(obs_dim=(1, ), action_dim=(1, )))
        policy = GaussianGRUPolicy(env_spec=env.spec,
                                   state_include_action=False)
        env.reset()
        obs = env.reset()[0]
        with tf.compat.v1.variable_scope('GaussianGRUPolicy', reuse=True):
            param = tf.compat.v1.get_variable(
                'dist_params/log_std_param/parameter')
        # assign it to all one
        param.load(tf.ones_like(param).eval())

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist
        output1 = self.sess.run(
            [dist_sym.loc, dist_sym.stddev()],
            feed_dict={state_input: [[obs.flatten()], [obs.flatten()]]})

        p = pickle.dumps(policy)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            # yapf: disable
            state_input = tf.compat.v1.placeholder(tf.float32,
                                                   shape=(None, None,
                                                          policy.input_dim))
            dist_sym = policy_pickled.build(state_input, name='dist_sym').dist
            output2 = sess.run(
                [
                    dist_sym.loc,
                    dist_sym.stddev()
                ],
                feed_dict={
                    state_input: [[obs.flatten()], [obs.flatten()]]
                })
            assert np.array_equal(output1, output2)
            # yapf: enable

    def test_state_info_specs(self):
        env = GymEnv(DummyBoxEnv(obs_dim=(4, ), action_dim=(4, )))
        policy = GaussianGRUPolicy(env_spec=env.spec,
                                   state_include_action=False)
        assert policy.state_info_specs == []

    def test_state_info_specs_with_state_include_action(self):
        env = GymEnv(DummyBoxEnv(obs_dim=(4, ), action_dim=(4, )))
        policy = GaussianGRUPolicy(env_spec=env.spec,
                                   state_include_action=True)
        assert policy.state_info_specs == [('prev_action', (4, ))]

    def test_clone(self):
        env = GymEnv(DummyBoxEnv(obs_dim=(4, ), action_dim=(4, )))
        policy = GaussianGRUPolicy(env_spec=env.spec)
        policy_clone = policy.clone('GaussianGRUPolicyClone')
        assert policy_clone.env_spec == policy.env_spec
        for cloned_param, param in zip(policy_clone.parameters.values(),
                                       policy.parameters.values()):
            assert np.array_equal(cloned_param, param)
