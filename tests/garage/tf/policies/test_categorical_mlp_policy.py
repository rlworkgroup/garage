import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv
from tests.fixtures.models import SimpleMLPModel


class TestCategoricalMLPPolicy(TfGraphTestCase):

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    @mock.patch('numpy.random.choice')
    def test_get_action(self, mock_rand, obs_dim, action_dim):
        mock_rand.return_value = 0
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'categorical_mlp_policy.MLPModel'),
                        new=SimpleMLPModel):
            policy = CategoricalMLPPolicy(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        action, prob = policy.get_action(obs)
        expected_prob = np.full(action_dim, 0.5)

        assert env.action_space.contains(action)
        assert action == 0
        assert np.array_equal(prob['prob'], expected_prob)

        actions, probs = policy.get_actions([obs, obs, obs])
        for action, prob in zip(actions, probs['prob']):
            assert env.action_space.contains(action)
            assert action == 0
            assert np.array_equal(prob, expected_prob)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_dist_info(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'categorical_mlp_policy.MLPModel'),
                        new=SimpleMLPModel):
            policy = CategoricalMLPPolicy(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        expected_prob = np.full(action_dim, 0.5)

        policy_probs = policy.dist_info([obs.flatten()])
        assert np.array_equal(policy_probs['prob'][0], expected_prob)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_dist_info_sym(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'categorical_mlp_policy.MLPModel'),
                        new=SimpleMLPModel):
            policy = CategoricalMLPPolicy(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        expected_prob = np.full(action_dim, 0.5)

        obs_dim = env.spec.observation_space.flat_dim
        state_input = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, obs_dim))
        dist1 = policy.dist_info_sym(state_input, name='policy2')

        prob = self.sess.run(dist1['prob'],
                             feed_dict={state_input: [obs.flatten()]})
        assert np.array_equal(prob[0], expected_prob)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_is_pickleable(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'categorical_mlp_policy.MLPModel'),
                        new=SimpleMLPModel):
            policy = CategoricalMLPPolicy(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        with tf.compat.v1.variable_scope('CategoricalMLPPolicy/MLPModel',
                                         reuse=True):
            return_var = tf.compat.v1.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())
        output1 = self.sess.run(
            policy.model.outputs,
            feed_dict={policy.model.input: [obs.flatten()]})

        p = pickle.dumps(policy)

        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            output2 = sess.run(
                policy_pickled.model.outputs,
                feed_dict={policy_pickled.model.input: [obs.flatten()]})
            assert np.array_equal(output1, output2)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), 1),
        ((2, ), 2),
        ((1, 1), 1),
        ((2, 2), 2),
    ])
    def test_get_regularizable_vars(self, obs_dim, action_dim):
        env = TfEnv(DummyDiscreteEnv(obs_dim=obs_dim, action_dim=action_dim))
        policy = CategoricalMLPPolicy(env_spec=env.spec)
        reg_vars = policy.get_regularizable_vars()
        assert len(reg_vars) == 2
        for var in reg_vars:
            assert ('bias' not in var.name) and ('output' not in var.name)
