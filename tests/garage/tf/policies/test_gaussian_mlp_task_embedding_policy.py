from itertools import chain
import pickle
from unittest import mock

# pylint: disable=wrong-import-order
import akro
import numpy as np
import pytest
import tensorflow as tf

from garage import InOutSpec
from garage.envs import GymEnv
from garage.tf.embeddings import GaussianMLPEncoder
from garage.tf.policies import GaussianMLPTaskEmbeddingPolicy

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestGaussianMLPTaskEmbeddingPolicy(TfGraphTestCase):

    @pytest.mark.parametrize('obs_dim', [(2, ), (2, 2)])
    @pytest.mark.parametrize('task_num', [1, 5])
    @pytest.mark.parametrize('latent_dim', [1, 5])
    @pytest.mark.parametrize('action_dim', [(2, ), (2, 2)])
    def test_get_action(self, obs_dim, task_num, latent_dim, action_dim):
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        embedding_spec = InOutSpec(
            input_space=akro.Box(low=np.zeros(task_num),
                                 high=np.ones(task_num)),
            output_space=akro.Box(low=np.zeros(latent_dim),
                                  high=np.ones(latent_dim)))
        encoder = GaussianMLPEncoder(embedding_spec)
        policy = GaussianMLPTaskEmbeddingPolicy(env_spec=env.spec,
                                                encoder=encoder)

        env.reset()
        obs = env.step(1).observation
        latent = np.random.random((latent_dim, ))
        task = np.zeros(task_num)
        task[0] = 1

        action1, _ = policy.get_action_given_latent(obs, latent)
        action2, _ = policy.get_action_given_task(obs, task)
        action3, _ = policy.get_action(np.concatenate([obs.flatten(), task]))

        assert env.action_space.contains(action1)
        assert env.action_space.contains(action2)
        assert env.action_space.contains(action3)

        obses, latents, tasks = [obs] * 3, [latent] * 3, [task] * 3
        aug_obses = [np.concatenate([obs.flatten(), task])] * 3
        action1n, _ = policy.get_actions_given_latents(obses, latents)
        action2n, _ = policy.get_actions_given_tasks(obses, tasks)
        action3n, _ = policy.get_actions(aug_obses)

        for action in chain(action1n, action2n, action3n):
            assert env.action_space.contains(action)

    def test_get_latent(self):
        obs_dim, action_dim, task_num, latent_dim = (2, ), (2, ), 5, 2
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
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

    # pylint: disable=no-member
    def test_auxiliary(self):
        obs_dim, action_dim, task_num, latent_dim = (2, ), (2, ), 2, 2
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
        embedding_spec = InOutSpec(
            input_space=akro.Box(low=np.zeros(task_num),
                                 high=np.ones(task_num)),
            output_space=akro.Box(low=np.zeros(latent_dim),
                                  high=np.ones(latent_dim)))
        encoder = GaussianMLPEncoder(embedding_spec)
        policy = GaussianMLPTaskEmbeddingPolicy(env_spec=env.spec,
                                                encoder=encoder)
        obs_input = tf.compat.v1.placeholder(tf.float32, shape=(None, None, 2))
        task_input = tf.compat.v1.placeholder(tf.float32,
                                              shape=(None, None, 2))
        networks = policy.build(obs_input, task_input)
        dist = networks[0].dist
        encoder_dist = networks[1].dist

        assert dist.loc.get_shape().as_list()[-1] == env.action_space.flat_dim
        assert policy.encoder == encoder
        assert policy.latent_space.flat_dim == latent_dim
        assert policy.task_space.flat_dim == task_num
        assert (policy.augmented_observation_space.flat_dim ==
                env.observation_space.flat_dim + task_num)
        assert encoder_dist.loc.get_shape().as_list()[-1] == latent_dim

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
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
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
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=action_dim))
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
