import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GarageEnv
from garage.tf.policies import CategoricalGRUPolicy

# yapf: disable
from tests.fixtures import TfGraphTestCase  # noqa: I202
from tests.fixtures.envs.dummy import (DummyBoxEnv,
                                       DummyDictEnv,
                                       DummyDiscreteEnv)

# yapf: enable


class TestCategoricalGRUPolicy(TfGraphTestCase):

    def test_invalid_env(self):
        env = GarageEnv(DummyBoxEnv())
        with pytest.raises(ValueError):
            CategoricalGRUPolicy(env_spec=env.spec)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim', [
        ((1, ), 1, 4),
        ((2, ), 2, 4),
        ((1, 1), 1, 4),
        ((2, 2), 2, 4),
    ])
    def test_get_action_state_include_action(self, obs_dim, action_dim,
                                             hidden_dim):
        env = GarageEnv(
            DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalGRUPolicy(env_spec=env.spec,
                                      hidden_dim=hidden_dim,
                                      state_include_action=True)
        policy.reset()
        obs = env.reset()

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
    def test_build_state_include_action(self, obs_dim, action_dim, hidden_dim):
        env = GarageEnv(
            DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalGRUPolicy(env_spec=env.spec,
                                      hidden_dim=hidden_dim,
                                      state_include_action=True)
        policy.reset(do_resets=None)
        obs = env.reset()

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist

        concat_obs = np.concatenate([obs.flatten(), np.zeros(action_dim)])
        output1 = self.sess.run(
            [policy.distribution.probs],
            feed_dict={policy.model.input: [[concat_obs], [concat_obs]]})
        output2 = self.sess.run(
            [dist_sym.probs],
            feed_dict={state_input: [[concat_obs], [concat_obs]]})
        assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim', [
        ((1, ), 1, 4),
        ((2, ), 2, 4),
        ((1, 1), 1, 4),
        ((2, 2), 2, 4),
    ])
    def test_build_state_not_include_action(self, obs_dim, action_dim,
                                            hidden_dim):
        env = GarageEnv(
            DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalGRUPolicy(env_spec=env.spec,
                                      hidden_dim=hidden_dim,
                                      state_include_action=False)
        policy.reset(do_resets=None)
        obs = env.reset()

        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, None,
                                                      policy.input_dim))
        dist_sym = policy.build(state_input, name='dist_sym').dist
        output1 = self.sess.run(
            [policy.distribution.probs],
            feed_dict={policy.model.input: [[obs.flatten()], [obs.flatten()]]})
        output2 = self.sess.run(
            [dist_sym.probs],
            feed_dict={state_input: [[obs.flatten()], [obs.flatten()]]})
        assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim, obs_type', [
        ((1, ), 1, 4, 'discrete'),
        ((2, ), 2, 4, 'discrete'),
        ((1, 1), 1, 4, 'discrete'),
        ((2, 2), 2, 4, 'discrete'),
        ((1, ), 1, 4, 'dict'),
    ])
    def test_get_action(self, obs_dim, action_dim, hidden_dim, obs_type):
        assert obs_type in ['discrete', 'dict']
        if obs_type == 'discrete':
            env = GarageEnv(
                DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        else:
            env = GarageEnv(
                DummyDictEnv(obs_space_type='box', act_space_type='discrete'))
        policy = CategoricalGRUPolicy(env_spec=env.spec,
                                      hidden_dim=hidden_dim,
                                      state_include_action=False)
        policy.reset(do_resets=None)
        obs = env.reset()

        if obs_type == 'discrete':
            obs = obs.flatten()

        action, _ = policy.get_action(obs)
        assert env.action_space.contains(action)

        actions, _ = policy.get_actions([obs])
        for action in actions:
            assert env.action_space.contains(action)

    def test_is_pickleable(self):
        env = GarageEnv(DummyDiscreteEnv(obs_dim=(1, ), action_dim=1))
        policy = CategoricalGRUPolicy(env_spec=env.spec,
                                      state_include_action=False)

        obs = env.reset()
        policy.model._gru_cell.weights[0].load(
            tf.ones_like(policy.model._gru_cell.weights[0]).eval())

        output1 = self.sess.run(
            [policy.distribution.probs],
            feed_dict={policy.model.input: [[obs.flatten()], [obs.flatten()]]})

        p = pickle.dumps(policy)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            # yapf: disable
            output2 = sess.run(
                [policy_pickled.distribution.probs],
                feed_dict={
                    policy_pickled.model.input: [[obs.flatten()],
                                                 [obs.flatten()]]
                })
            # yapf: enable
            assert np.array_equal(output1, output2)

    def test_state_info_specs(self):
        env = GarageEnv(DummyDiscreteEnv(obs_dim=(10, ), action_dim=4))
        policy = CategoricalGRUPolicy(env_spec=env.spec,
                                      state_include_action=False)
        assert policy.state_info_specs == []

    def test_state_info_specs_with_state_include_action(self):
        env = GarageEnv(DummyDiscreteEnv(obs_dim=(10, ), action_dim=4))
        policy = CategoricalGRUPolicy(env_spec=env.spec,
                                      state_include_action=True)
        assert policy.state_info_specs == [('prev_action', (4, ))]
