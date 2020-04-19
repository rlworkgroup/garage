from itertools import chain
import pickle
from unittest import mock

import akro
import numpy as np
import pytest
import tensorflow as tf

from garage import InOutSpec
from garage.tf.embeddings import GaussianMLPEncoder
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPTaskEmbeddingPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.models import SimpleGaussianMLPModel


class TestGaussianMLPTaskEmbeddingPolicy(TfGraphTestCase):

    @pytest.mark.parametrize('obs_dim', [(2, ), (2, 2)])
    @pytest.mark.parametrize('task_num', [1, 5])
    @pytest.mark.parametrize('latent_dim', [1, 5])
    @pytest.mark.parametrize('action_dim', [(2, ), (2, 2)])
    @mock.patch('numpy.random.normal')
    def test_get_action(self, mock_normal, obs_dim, task_num, latent_dim,
                        action_dim):
        mock_normal.return_value = 0.5
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(
                'garage.tf.policies.'
                'gaussian_mlp_task_embedding_policy.GaussianMLPModel',
                new=SimpleGaussianMLPModel):
            embedding_spec = InOutSpec(
                input_space=akro.Box(low=np.zeros(task_num),
                                     high=np.ones(task_num)),
                output_space=akro.Box(low=np.zeros(latent_dim),
                                      high=np.ones(latent_dim)))
            encoder = GaussianMLPEncoder(embedding_spec)
            policy = GaussianMLPTaskEmbeddingPolicy(env_spec=env.spec,
                                                    encoder=encoder)

        env.reset()
        obs, _, _, _ = env.step(1)
        latent = np.random.random((latent_dim, ))
        task = np.zeros(task_num)
        task[0] = 1

        action1, prob1 = policy.get_action_given_latent(obs, latent)
        action2, prob2 = policy.get_action_given_task(obs, task)
        action3, prob3 = policy.get_action(
            np.concatenate([obs.flatten(), task]))

        expected_action = np.full(action_dim, 0.75)
        expected_mean = np.full(action_dim, 0.5)
        expected_log_std = np.full(action_dim, np.log(0.5))

        assert env.action_space.contains(action1)
        assert np.array_equal(action1, expected_action)
        assert np.array_equal(prob1['mean'], expected_mean)
        assert np.array_equal(prob1['log_std'], expected_log_std)

        assert env.action_space.contains(action2)
        assert np.array_equal(action2, expected_action)
        assert np.array_equal(prob2['mean'], expected_mean)
        assert np.array_equal(prob2['log_std'], expected_log_std)

        assert env.action_space.contains(action3)
        assert np.array_equal(action3, expected_action)
        assert np.array_equal(prob3['mean'], expected_mean)
        assert np.array_equal(prob3['log_std'], expected_log_std)

        obses, latents, tasks = [obs] * 3, [latent] * 3, [task] * 3
        aug_obses = [np.concatenate([obs.flatten(), task])] * 3
        action1n, prob1n = policy.get_actions_given_latents(obses, latents)
        action2n, prob2n = policy.get_actions_given_tasks(obses, tasks)
        action3n, prob3n = policy.get_actions(aug_obses)

        for action, mean, log_std in chain(
                zip(action1n, prob1n['mean'], prob1n['log_std']),
                zip(action2n, prob2n['mean'], prob2n['log_std']),
                zip(action3n, prob3n['mean'], prob3n['log_std'])):
            assert env.action_space.contains(action)
            assert np.array_equal(action, expected_action)
            assert np.array_equal(mean, expected_mean)
            assert np.array_equal(log_std, expected_log_std)

    def test_get_latent(self):
        obs_dim, action_dim, task_num, latent_dim = (2, ), (2, ), 5, 2
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(
                'garage.tf.policies.'
                'gaussian_mlp_task_embedding_policy.GaussianMLPModel',
                new=SimpleGaussianMLPModel):
            embedding_spec = InOutSpec(
                input_space=akro.Box(low=np.zeros(task_num),
                                     high=np.ones(task_num)),
                output_space=akro.Box(low=np.zeros(latent_dim),
                                      high=np.ones(latent_dim)))
            encoder = GaussianMLPEncoder(embedding_spec)
            policy = GaussianMLPTaskEmbeddingPolicy(env_spec=env.spec,
                                                    encoder=encoder)

            task_id = 3
            task_onehot = np.zeros(task_num)
            task_onehot[task_id] = 1
            latent, latent_info = policy.get_latent(task_onehot)
            assert latent.shape == (latent_dim, )
            assert latent_info['mean'].shape == (latent_dim, )
            assert latent_info['log_std'].shape == (latent_dim, )

    @pytest.mark.parametrize('obs_dim', [(2, ), (2, 2)])
    @pytest.mark.parametrize('task_num', [1, 5])
    @pytest.mark.parametrize('latent_dim', [1, 5])
    @pytest.mark.parametrize('action_dim', [(2, ), (2, 2)])
    def test_dist_info_sym(self, obs_dim, task_num, latent_dim, action_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(
                'garage.tf.policies.'
                'gaussian_mlp_task_embedding_policy.GaussianMLPModel',
                new=SimpleGaussianMLPModel):
            embedding_spec = InOutSpec(
                input_space=akro.Box(low=np.zeros(task_num),
                                     high=np.ones(task_num)),
                output_space=akro.Box(low=np.zeros(latent_dim),
                                      high=np.ones(latent_dim)))
            encoder = GaussianMLPEncoder(embedding_spec)
            policy = GaussianMLPTaskEmbeddingPolicy(env_spec=env.spec,
                                                    encoder=encoder)

        env.reset()
        obs, _, _, _ = env.step(1)
        task = np.zeros(task_num)
        task[0] = 1
        latent = np.random.random(latent_dim)

        obs_dim = env.spec.observation_space.flat_dim
        obs_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, obs_dim))
        task_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, task_num))
        latent_ph = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, latent_dim))

        dist1_sym = policy.dist_info_sym_given_task(obs_ph,
                                                    task_ph,
                                                    name='p1_sym')
        dist2_sym = policy.dist_info_sym_given_latent(obs_ph,
                                                      latent_ph,
                                                      name='p2_sym')

        # flatten output
        expected_mean = [np.full(np.prod(action_dim), 0.5)]
        expected_log_std = [np.full(np.prod(action_dim), np.log(0.5))]

        prob1 = self.sess.run(dist1_sym,
                              feed_dict={
                                  obs_ph: [obs.flatten()],
                                  task_ph: [task]
                              })
        prob2 = self.sess.run(dist2_sym,
                              feed_dict={
                                  obs_ph: [obs.flatten()],
                                  latent_ph: [latent]
                              })

        assert np.array_equal(prob1['mean'], expected_mean)
        assert np.array_equal(prob1['log_std'], expected_log_std)
        assert np.array_equal(prob2['mean'], expected_mean)
        assert np.array_equal(prob2['log_std'], expected_log_std)

    def test_encoder_dist_info(self):
        obs_dim, action_dim, task_num, latent_dim = (2, ), (2, ), 5, 2
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(
                'garage.tf.embeddings.'
                'gaussian_mlp_encoder.GaussianMLPModel',
                new=SimpleGaussianMLPModel):

            old_build = SimpleGaussianMLPModel._build

            def float32_build(this, obs_input, name):
                mean, log_std, std, dist = old_build(this, obs_input, name)
                return mean, tf.cast(log_std, tf.float32), std, dist

            SimpleGaussianMLPModel._build = float32_build

            embedding_spec = InOutSpec(
                input_space=akro.Box(low=np.zeros(task_num),
                                     high=np.ones(task_num)),
                output_space=akro.Box(low=np.zeros(latent_dim),
                                      high=np.ones(latent_dim)))
            encoder = GaussianMLPEncoder(embedding_spec)
            policy = GaussianMLPTaskEmbeddingPolicy(env_spec=env.spec,
                                                    encoder=encoder)

            assert policy.encoder_distribution.dim == latent_dim

            inp_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, 5))
            dist_sym = policy.encoder_dist_info_sym(inp_ph)
            dist = self.sess.run(dist_sym,
                                 feed_dict={inp_ph: [np.random.random(5)]})

            expected_mean = np.full(latent_dim, 0.5)
            expected_log_std = np.full(latent_dim, np.log(0.5))

            assert np.allclose(dist['mean'], expected_mean)
            assert np.allclose(dist['log_std'], expected_log_std)

            SimpleGaussianMLPModel._dtype = np.float32

    def test_auxiliary(self):
        obs_dim, action_dim, task_num, latent_dim = (2, ), (2, ), 2, 2
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        with mock.patch(
                'garage.tf.policies.'
                'gaussian_mlp_task_embedding_policy.GaussianMLPModel',
                new=SimpleGaussianMLPModel):
            embedding_spec = InOutSpec(
                input_space=akro.Box(low=np.zeros(task_num),
                                     high=np.ones(task_num)),
                output_space=akro.Box(low=np.zeros(latent_dim),
                                      high=np.ones(latent_dim)))
            encoder = GaussianMLPEncoder(embedding_spec)
            policy = GaussianMLPTaskEmbeddingPolicy(env_spec=env.spec,
                                                    encoder=encoder)

        assert policy.distribution.dim == env.action_space.flat_dim
        assert policy.encoder == encoder
        assert policy.latent_space.flat_dim == latent_dim
        assert policy.task_space.flat_dim == task_num
        assert (policy.augmented_observation_space.flat_dim ==
                env.observation_space.flat_dim + task_num)
        assert policy.encoder_distribution.dim == latent_dim

    def test_split_augmented_observation(self):
        obs_dim, task_num = 3, 5
        policy = mock.Mock(spec=GaussianMLPTaskEmbeddingPolicy)
        policy.task_space = mock.Mock()
        policy.task_space.flat_dim = task_num
        policy.split_augmented_observation = \
            GaussianMLPTaskEmbeddingPolicy.split_augmented_observation

        obs = np.random.random(obs_dim)
        task = np.random.random(task_num)
        o, t = policy.split_augmented_observation(policy,
                                                  np.concatenate([obs, task]))

        assert np.array_equal(obs, o)
        assert np.array_equal(task, t)

    def test_get_vars(self):
        obs_dim, action_dim, task_num, latent_dim = (2, ), (2, ), 5, 2
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        embedding_spec = InOutSpec(
            input_space=akro.Box(low=np.zeros(task_num),
                                 high=np.ones(task_num)),
            output_space=akro.Box(low=np.zeros(latent_dim),
                                  high=np.ones(latent_dim)))
        encoder = GaussianMLPEncoder(embedding_spec, hidden_sizes=[32, 32, 32])
        policy = GaussianMLPTaskEmbeddingPolicy(env_spec=env.spec,
                                                encoder=encoder,
                                                hidden_sizes=[32, 32, 32])

        vars1 = sorted(policy.get_trainable_vars(), key=lambda v: v.name)
        vars2 = sorted(policy.get_global_vars(), key=lambda v: v.name)
        assert vars1 == vars2

        # Two network. Each with 4 layers * (1 weight + 1 bias) + 1 log_std
        assert len(vars1) == 2 * (4 * 2 + 1)

        obs = np.random.random(obs_dim)
        latent = np.random.random((latent_dim, ))

        for var in vars1:
            var.assign(np.ones(var.shape))
        assert np.any(policy.get_action_given_latent(obs, latent) != 0)

        for var in vars1:
            var.assign(np.zeros(var.shape))
        assert not np.all(policy.get_action_given_latent(obs, latent) == 0)

    def test_pickling(self):
        obs_dim, action_dim, task_num, latent_dim = (2, ), (2, ), 5, 2
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        embedding_spec = InOutSpec(
            input_space=akro.Box(low=np.zeros(task_num),
                                 high=np.ones(task_num)),
            output_space=akro.Box(low=np.zeros(latent_dim),
                                  high=np.ones(latent_dim)))
        encoder = GaussianMLPEncoder(embedding_spec)
        policy = GaussianMLPTaskEmbeddingPolicy(env_spec=env.spec,
                                                encoder=encoder)

        pickled = pickle.dumps(policy)
        with tf.compat.v1.variable_scope('resumed'):
            unpickled = pickle.loads(pickled)
            assert hasattr(unpickled, '_f_dist_obs_latent')
            assert hasattr(unpickled, '_f_dist_obs_task')
