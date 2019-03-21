import pickle
from unittest import mock

from nose2.tools.params import params
import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.models import SimpleGaussianMLPModel


class TestGaussianMLPPolicyWithModel(TfGraphTestCase):
    @params(
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    )
    def test_get_action(self, obs_dim, action_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'gaussian_mlp_policy_with_model.GaussianMLPModel'),
                        new=SimpleGaussianMLPModel):
            policy = GaussianMLPPolicyWithModel(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        action, prob = policy.get_action(obs)

        expected_action = np.full(action_dim, 0.75)
        expected_mean = np.full(action_dim, 0.5)
        expected_log_std = np.full(action_dim, 0.5)

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

    @params(
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    )
    def test_dist_info_sym(self, obs_dim, action_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'gaussian_mlp_policy_with_model.GaussianMLPModel'),
                        new=SimpleGaussianMLPModel):
            policy = GaussianMLPPolicyWithModel(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        obs_dim = env.spec.observation_space.flat_dim
        obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))

        dist1_sym = policy.dist_info_sym(obs_ph, name='p1_sym')

        expected_mean = np.full(action_dim, 0.5)
        expected_log_std = np.full(action_dim, 0.5)

        prob = self.sess.run(dist1_sym, feed_dict={obs_ph: [obs.flatten()]})

        assert np.array_equal(prob['mean'], expected_mean)
        assert np.array_equal(prob['log_std'], expected_log_std)

    @params(
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    )
    def test_is_pickleable(self, obs_dim, action_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'gaussian_mlp_policy_with_model.GaussianMLPModel'),
                        new=SimpleGaussianMLPModel):
            policy = GaussianMLPPolicyWithModel(env_spec=env.spec)

        env.reset()
        obs, _, _, _ = env.step(1)
        obs_dim = env.spec.observation_space.flat_dim

        action1, prob1 = policy.get_action(obs)

        p = pickle.dumps(policy)
        with tf.Session(graph=tf.Graph()):
            policy_pickled = pickle.loads(p)
            action2, prob2 = policy_pickled.get_action(obs)

        assert env.action_space.contains(action1)
        assert np.array_equal(action1, action2)
        assert np.array_equal(prob1['mean'], prob2['mean'])
        assert np.array_equal(prob1['log_std'], prob2['log_std'])
