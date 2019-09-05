import pickle
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.models import SimpleGaussianMLPModel


class TestGaussianMLPPolicy(TfGraphTestCase):

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    @mock.patch('numpy.random.normal')
    def test_get_action(self, mock_normal, obs_dim, action_dim):
        mock_normal.return_value = 0.5
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'gaussian_mlp_policy.GaussianMLPModel'),
                        new=SimpleGaussianMLPModel):
            policy = GaussianMLPPolicy(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        action, prob = policy.get_action(obs)

        expected_action = np.full(action_dim, 0.75)
        expected_mean = np.full(action_dim, 0.5)
        expected_log_std = np.full(action_dim, np.log(0.5))

        assert env.action_space.contains(action)
        assert np.array_equal(action, expected_action)
        assert np.array_equal(prob['mean'], expected_mean)
        assert np.array_equal(prob['log_std'], expected_log_std)

        actions, probs = policy.get_actions([obs, obs, obs])
        for action, mean, log_std in zip(actions, probs['mean'],
                                         probs['log_std']):
            assert env.action_space.contains(action)
            assert np.array_equal(action, expected_action)
            assert np.array_equal(prob['mean'], expected_mean)
            assert np.array_equal(prob['log_std'], expected_log_std)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_dist_info_sym(self, obs_dim, action_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'gaussian_mlp_policy.GaussianMLPModel'),
                        new=SimpleGaussianMLPModel):
            policy = GaussianMLPPolicy(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        obs_dim = env.spec.observation_space.flat_dim
        obs_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, obs_dim))

        dist1_sym = policy.dist_info_sym(obs_ph, name='p1_sym')

        # flatten output
        expected_mean = [np.full(np.prod(action_dim), 0.5)]
        expected_log_std = [np.full(np.prod(action_dim), np.log(0.5))]

        prob = self.sess.run(dist1_sym, feed_dict={obs_ph: [obs.flatten()]})

        assert np.array_equal(prob['mean'], expected_mean)
        assert np.array_equal(prob['log_std'], expected_log_std)

    @pytest.mark.parametrize('obs_dim, action_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_is_pickleable(self, obs_dim, action_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'gaussian_mlp_policy.GaussianMLPModel'),
                        new=SimpleGaussianMLPModel):
            policy = GaussianMLPPolicy(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)
        obs_dim = env.spec.observation_space.flat_dim

        with tf.compat.v1.variable_scope('GaussianMLPPolicy/GaussianMLPModel',
                                         reuse=True):
            return_var = tf.compat.v1.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())
        output1 = self.sess.run(
            policy.model.outputs[:-1],
            feed_dict={policy.model.input: [obs.flatten()]})

        p = pickle.dumps(policy)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            output2 = sess.run(
                policy_pickled.model.outputs[:-1],
                feed_dict={policy_pickled.model.input: [obs.flatten()]})
            assert np.array_equal(output1, output2)
