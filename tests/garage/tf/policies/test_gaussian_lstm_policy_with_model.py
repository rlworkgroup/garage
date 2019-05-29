import pickle
from unittest import mock

from nose2.tools.params import params
import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianLSTMPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.envs.dummy import DummyDiscreteEnv
from tests.fixtures.models import SimpleGaussianLSTMModel


class TestGaussianLSTMPolicyWithModel(TfGraphTestCase):
    @params(
        ((1, ), (1, ), 4),
        ((2, ), (2, ), 4),
        ((1, 1), (1, ), 4),
        ((2, 2), (2, ), 4),
    )
    def test_dist_info_sym(self, obs_dim, action_dim, hidden_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))

        obs_ph = tf.placeholder(
            tf.float32, shape=(None, None, env.observation_space.flat_dim))

        with mock.patch(('garage.tf.policies.'
                         'gaussian_lstm_policy_with_model.GaussianLSTMModel'),
                        new=SimpleGaussianLSTMModel):
            policy = GaussianLSTMPolicyWithModel(
                env_spec=env.spec, state_include_action=False)

            policy.reset()
            obs = env.reset()

            dist_sym = policy.dist_info_sym(
                obs_var=obs_ph, state_info_vars=None, name='p2_sym')

        dist = self.sess.run(
            dist_sym, feed_dict={obs_ph: [[obs.flatten()], [obs.flatten()]]})

        assert np.array_equal(dist['mean'], np.full((2, 1) + action_dim, 0.5))
        assert np.array_equal(dist['log_std'], np.full((2, 1) + action_dim,
                                                       0.5))

    @params(
        ((1, ), (1, ), 4),
        ((2, ), (2, ), 4),
        ((1, 1), (1, ), 4),
        ((2, 2), (2, ), 4),
    )
    def test_dist_info_sym_include_action(self, obs_dim, action_dim,
                                          hidden_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))

        obs_ph = tf.placeholder(
            tf.float32, shape=(None, None, env.observation_space.flat_dim))

        with mock.patch(('garage.tf.policies.'
                         'gaussian_lstm_policy_with_model.GaussianLSTMModel'),
                        new=SimpleGaussianLSTMModel):
            policy = GaussianLSTMPolicyWithModel(
                env_spec=env.spec, state_include_action=True)

            policy.reset()
            obs = env.reset()
            dist_sym = policy.dist_info_sym(
                obs_var=obs_ph,
                state_info_vars={'prev_action': np.zeros((2, 1) + action_dim)},
                name='p2_sym')
        dist = self.sess.run(
            dist_sym, feed_dict={obs_ph: [[obs.flatten()], [obs.flatten()]]})

        assert np.array_equal(dist['mean'], np.full((2, 1) + action_dim, 0.5))
        assert np.array_equal(dist['log_std'], np.full((2, 1) + action_dim,
                                                       0.5))

    def test_dist_info_sym_wrong_input(self):
        env = TfEnv(DummyBoxEnv(obs_dim=(1, ), action_dim=(1, )))

        obs_ph = tf.placeholder(
            tf.float32, shape=(None, None, env.observation_space.flat_dim))

        with mock.patch(('garage.tf.policies.'
                         'gaussian_lstm_policy_with_model.GaussianLSTMModel'),
                        new=SimpleGaussianLSTMModel):
            policy = GaussianLSTMPolicyWithModel(
                env_spec=env.spec, state_include_action=True)

            policy.reset()
            obs = env.reset()

            policy.dist_info_sym(
                obs_var=obs_ph,
                state_info_vars={'prev_action': np.zeros((3, 1, 1))},
                name='p2_sym')
        # observation batch size = 2 but prev_action batch size = 3
        with self.assertRaises(tf.errors.InvalidArgumentError):
            self.sess.run(
                policy.model.networks['p2_sym'].input,
                feed_dict={obs_ph: [[obs.flatten()], [obs.flatten()]]})

    def test_invalid_env(self):
        env = TfEnv(DummyDiscreteEnv())
        with self.assertRaises(ValueError):
            GaussianLSTMPolicyWithModel(env_spec=env.spec)

    # yapf: disable
    @params(
        ((1, ), (1, ), 4),
        ((2, ), (2, ), 4),
        ((1, 1), (1, ), 4),
        ((2, 2), (2, ), 4))
    # yapf: enable
    def test_get_action_state_include_action(self, obs_dim, action_dim,
                                             hidden_dim):

        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'gaussian_lstm_policy_with_model.GaussianLSTMModel'),
                        new=SimpleGaussianLSTMModel):
            policy = GaussianLSTMPolicyWithModel(
                env_spec=env.spec, state_include_action=True)

        policy.reset()
        obs = env.reset()

        action, agent_info = policy.get_action(obs)
        assert env.action_space.contains(action)
        assert np.array_equal(action, np.full(action_dim, 0.75))
        expected_mean = np.full(action_dim, 0.5)
        assert np.array_equal(agent_info['mean'], expected_mean)
        expected_log_std = np.full(action_dim, 0.5)
        assert np.array_equal(agent_info['log_std'], expected_log_std)
        expected_prev_action = np.full(action_dim, 0)
        assert np.array_equal(agent_info['prev_action'], expected_prev_action)

        policy.reset()

        actions, agent_infos = policy.get_actions([obs])
        for action, mean, log_std, prev_action in zip(
                actions, agent_infos['mean'], agent_infos['log_std'],
                agent_infos['prev_action']):
            assert env.action_space.contains(action)
            assert np.array_equal(action, np.full(action_dim, 0.75))
            assert np.array_equal(mean, expected_mean)
            assert np.array_equal(log_std, expected_log_std)
            assert np.array_equal(prev_action, expected_prev_action)

    # yapf: disable
    @params(
        ((1, ), (1, ), 4),
        ((2, ), (2, ), 4),
        ((1, 1), (1, ), 4),
        ((2, 2), (2, ), 4))
    # yapf: enable
    def test_get_action(self, obs_dim, action_dim, hidden_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(('garage.tf.policies.'
                         'gaussian_lstm_policy_with_model.GaussianLSTMModel'),
                        new=SimpleGaussianLSTMModel):
            policy = GaussianLSTMPolicyWithModel(
                env_spec=env.spec, state_include_action=False)

        policy.reset()
        obs = env.reset()

        action, agent_info = policy.get_action(obs)
        assert env.action_space.contains(action)
        assert np.array_equal(action, np.full(action_dim, 0.75))
        expected_mean = np.full(action_dim, 0.5)
        assert np.array_equal(agent_info['mean'], expected_mean)
        expected_log_std = np.full(action_dim, 0.5)
        assert np.array_equal(agent_info['log_std'], expected_log_std)

        actions, agent_infos = policy.get_actions([obs])
        for action, mean, log_std in zip(actions, agent_infos['mean'],
                                         agent_infos['log_std']):
            assert env.action_space.contains(action)
            assert np.array_equal(action, np.full(action_dim, 0.75))
            assert np.array_equal(mean, expected_mean)
            assert np.array_equal(log_std, expected_log_std)

    def test_is_pickleable(self):
        env = TfEnv(DummyBoxEnv(obs_dim=(1, ), action_dim=(1, )))
        with mock.patch(('garage.tf.policies.'
                         'gaussian_lstm_policy_with_model.GaussianLSTMModel'),
                        new=SimpleGaussianLSTMModel):
            policy = GaussianLSTMPolicyWithModel(
                env_spec=env.spec, state_include_action=False)

        env.reset()
        obs = env.reset()

        with tf.variable_scope(
                'GaussianLSTMPolicyWithModel/GaussianLSTMModel', reuse=True):
            return_var = tf.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())

        output1 = self.sess.run(
            policy.model.networks['default'].mean,
            feed_dict={policy.model.input: [[obs.flatten()], [obs.flatten()]]})

        p = pickle.dumps(policy)

        with tf.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            output2 = sess.run(
                policy_pickled.model.networks['default'].mean,
                feed_dict={
                    policy_pickled.model.input: [[obs.flatten()],
                                                 [obs.flatten()]]
                })
            assert np.array_equal(output1, output2)
