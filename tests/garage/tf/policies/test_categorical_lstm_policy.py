import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GymEnv
from garage.tf.policies import CategoricalLSTMPolicy

# yapf: disable
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import (DummyBoxEnv,
                                       DummyDictEnv,
                                       DummyDiscreteEnv)

# yapf: enable


class TestCategoricalLSTMPolicy(TfGraphTestCase):

    def test_invalid_env(self):
        env = GymEnv(DummyBoxEnv())
        with pytest.raises(ValueError):
            CategoricalLSTMPolicy(env_spec=env.spec)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim, obs_type', [
        ((1, ), 1, 4, 'discrete'),
        ((2, ), 2, 4, 'discrete'),
        ((1, 1), 1, 4, 'discrete'),
        ((2, 2), 2, 4, 'discrete'),
        ((1, ), 1, 4, 'dict'),
    ])
    def test_get_action_state_include_action(self, obs_dim, action_dim,
                                             hidden_dim, obs_type):
        assert obs_type in ['discrete', 'dict']
        if obs_type == 'discrete':
            env = GymEnv(
                DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        else:
            env = GymEnv(
                DummyDictEnv(obs_space_type='box', act_space_type='discrete'))
        policy = CategoricalLSTMPolicy(env_spec=env.spec,
                                       hidden_dim=hidden_dim,
                                       state_include_action=True)

        policy.reset()
        obs = env.reset()[0]
        if obs_type == 'discrete':
            obs = obs.flatten()

        action, _ = policy.get_action(obs)
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs])
        for action in actions:
            assert env.action_space.contains(action)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim', [
        ((1, ), 1, 4),
        ((2, ), 2, 4),
        ((1, 1), 1, 4),
        ((2, 2), 2, 4),
    ])
    def test_get_action(self, obs_dim, action_dim, hidden_dim):
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalLSTMPolicy(env_spec=env.spec,
                                       hidden_dim=hidden_dim,
                                       state_include_action=False)

        policy.reset()
        obs = env.reset()[0]

        action, _ = policy.get_action(obs.flatten())
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs.flatten()])
        for action in actions:
            assert env.action_space.contains(action)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim', [
        ((1, ), 1, 4),
        ((2, ), 2, 4),
        ((1, 1), 1, 4),
        ((2, 2), 2, 4),
    ])
    # pylint: disable=no-member
    def test_build_state_include_action(self, obs_dim, action_dim, hidden_dim):
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalLSTMPolicy(env_spec=env.spec,
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
            [dist_sym.probs],
            feed_dict={state_input: [[concat_obs], [concat_obs]]})
        output2 = self.sess.run(
            [dist_sym2.probs],
            feed_dict={state_input: [[concat_obs], [concat_obs]]})
        assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim', [
        ((1, ), 1, 4),
        ((2, ), 2, 4),
        ((1, 1), 1, 4),
        ((2, 2), 2, 4),
    ])
    # pylint: disable=no-member
    def test_build_state_not_include_action(self, obs_dim, action_dim,
                                            hidden_dim):
        env = GymEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalLSTMPolicy(env_spec=env.spec,
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
            [dist_sym.probs],
            feed_dict={state_input: [[obs.flatten()], [obs.flatten()]]})
        output2 = self.sess.run(
            [dist_sym2.probs],
            feed_dict={state_input: [[obs.flatten()], [obs.flatten()]]})
        assert np.array_equal(output1, output2)

    # pylint: disable=no-member
    def test_is_pickleable(self):
        env = GymEnv(DummyDiscreteEnv(obs_dim=(1, ), action_dim=1))
        policy = CategoricalLSTMPolicy(env_spec=env.spec,
                                       state_include_action=False)

        policy.reset()
        obs = env.reset()[0]

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist
        policy._lstm_cell.weights[0].load(
            tf.ones_like(policy._lstm_cell.weights[0]).eval())

        output1 = self.sess.run(
            [dist_sym.probs],
            feed_dict={state_input: [[obs.flatten()], [obs.flatten()]]})

        p = pickle.dumps(policy)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            state_input = tf.compat.v1.placeholder(
                tf.float32, shape=(None, None, policy_pickled.input_dim))
            dist_sym = policy_pickled.build(state_input, name='dist_sym').dist
            output2 = sess.run(
                [dist_sym.probs],
                feed_dict={state_input: [[obs.flatten()],
                                         [obs.flatten()]]})  # noqa: E126
            assert np.array_equal(output1, output2)

    def test_state_info_specs(self):
        env = GymEnv(DummyDiscreteEnv(obs_dim=(10, ), action_dim=4))
        policy = CategoricalLSTMPolicy(env_spec=env.spec,
                                       state_include_action=False)
        assert policy.state_info_specs == []

    def test_state_info_specs_with_state_include_action(self):
        env = GymEnv(DummyDiscreteEnv(obs_dim=(10, ), action_dim=4))
        policy = CategoricalLSTMPolicy(env_spec=env.spec,
                                       state_include_action=True)
        assert policy.state_info_specs == [('prev_action', (4, ))]

    def test_clone(self):
        env = GymEnv(DummyDiscreteEnv(obs_dim=(10, ), action_dim=4))
        policy = CategoricalLSTMPolicy(env_spec=env.spec)
        policy_clone = policy.clone('CategoricalLSTMPolicyClone')
        assert policy.env_spec == policy_clone.env_spec
        for cloned_param, param in zip(policy_clone.parameters.values(),
                                       policy.parameters.values()):
            assert np.array_equal(cloned_param, param)
