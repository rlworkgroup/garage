import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalLSTMPolicy2
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.envs.dummy import DummyDiscreteEnv
from tests.fixtures.models import SimpleCategoricalLSTMModel


class TestCategoricalLSTMPolicy2(TfGraphTestCase):

    def test_invalid_env(self):
        env = TfEnv(DummyBoxEnv())
        with pytest.raises(ValueError):
            CategoricalLSTMPolicy2(env_spec=env.spec)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim', [
        ((1, ), 1, 4),
        ((2, ), 2, 4),
        ((1, 1), 1, 4),
        ((2, 2), 2, 4),
    ])
    @mock.patch('numpy.random.choice')
    def test_get_action_state_include_action(self, mock_rand, obs_dim,
                                             action_dim, hidden_dim):
        mock_rand.return_value = 0

        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        obs_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, None, env.observation_space.flat_dim + action_dim],
            name='obs')
        with mock.patch(('garage.tf.policies.'
                         'categorical_lstm_policy2.CategoricalLSTMModel'),
                        new=SimpleCategoricalLSTMModel):
            policy = CategoricalLSTMPolicy2(env_spec=env.spec,
                                            hidden_dim=hidden_dim,
                                            state_include_action=True)

        policy.build(obs_var)
        policy.reset()
        obs = env.reset()

        expected_prob = np.full(action_dim, 0.5)

        action, agent_info = policy.get_action(obs.flatten())
        assert env.action_space.contains(action)
        assert action == 0
        assert np.array_equal(agent_info['prob'], expected_prob)

        actions, agent_infos = policy.get_actions([obs.flatten()])
        for action, prob in zip(actions, agent_infos['prob']):
            assert env.action_space.contains(action)
            assert action == 0
            assert np.array_equal(prob, expected_prob)

    @pytest.mark.parametrize('obs_dim, action_dim, hidden_dim', [
        ((1, ), 1, 4),
        ((2, ), 2, 4),
        ((1, 1), 1, 4),
        ((2, 2), 2, 4),
    ])
    @mock.patch('numpy.random.choice')
    def test_get_action(self, mock_rand, obs_dim, action_dim, hidden_dim):
        mock_rand.return_value = 0

        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        obs_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, None, env.observation_space.flat_dim],
            name='obs')
        with mock.patch(('garage.tf.policies.'
                         'categorical_lstm_policy2.CategoricalLSTMModel'),
                        new=SimpleCategoricalLSTMModel):
            policy = CategoricalLSTMPolicy2(env_spec=env.spec,
                                            hidden_dim=hidden_dim,
                                            state_include_action=False)

        policy.build(obs_var)
        policy.reset()
        obs = env.reset()

        expected_prob = np.full(action_dim, 0.5)

        action, agent_info = policy.get_action(obs.flatten())
        assert env.action_space.contains(action)
        assert action == 0
        assert np.array_equal(agent_info['prob'], expected_prob)

        actions, agent_infos = policy.get_actions([obs.flatten()])
        for action, prob in zip(actions, agent_infos['prob']):
            assert env.action_space.contains(action)
            assert action == 0
            assert np.array_equal(prob, expected_prob)

    def test_is_pickleable(self):
        env = TfEnv(DummyDiscreteEnv(obs_dim=(1, ), action_dim=1))
        obs_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=[None, None, env.observation_space.flat_dim],
            name='obs')
        with mock.patch(('garage.tf.policies.'
                         'categorical_lstm_policy2.CategoricalLSTMModel'),
                        new=SimpleCategoricalLSTMModel):
            policy = CategoricalLSTMPolicy2(env_spec=env.spec,
                                            state_include_action=False)

        policy.build(obs_var)
        env.reset()
        obs = env.reset()

        with tf.compat.v1.variable_scope('CategoricalLSTMPolicy/prob_network',
                                         reuse=True):
            return_var = tf.compat.v1.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())

        output1 = self.sess.run(
            policy.model.outputs[0],
            feed_dict={policy.model.input: [[obs.flatten()], [obs.flatten()]]})

        p = pickle.dumps(policy)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            obs_var = tf.compat.v1.placeholder(
                tf.float32,
                shape=[None, None, env.observation_space.flat_dim],
                name='obs')
            policy_pickled.build(obs_var)
            output2 = sess.run(policy_pickled.model.outputs[0],
                               feed_dict={
                                   policy_pickled.model.input:
                                   [[obs.flatten()], [obs.flatten()]]
                               })  # noqa: E126
            assert np.array_equal(output1, output2)
