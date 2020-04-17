import pickle
from unittest import mock

import akro
import numpy as np
import pytest
import tensorflow as tf

from garage.tf.embeddings import EmbeddingSpec, GaussianMLPEmbedding
from garage.tf.embeddings.utils import concat_spaces
from garage.tf.envs import TfEnv
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.models import SimpleGaussianMLPModel


class TestGaussianMLPEmbedding(TfGraphTestCase):

    @pytest.mark.parametrize('obs_dim, embedding_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    @mock.patch('numpy.random.normal')
    def test_get_embedding(self, mock_normal, obs_dim, embedding_dim):
        mock_normal.return_value = 0.5
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=embedding_dim))
        with mock.patch(('garage.tf.embeddings.'
                         'gaussian_mlp_embedding.GaussianMLPModel'),
                        new=SimpleGaussianMLPModel):
            embedding_spec = EmbeddingSpec(
                input_space=env.spec.observation_space,
                latent_space=env.spec.action_space)
            embedding = GaussianMLPEmbedding(embedding_spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        latent, prob = embedding.get_latent(obs)

        expected_embedding = np.full(embedding_dim, 0.75)
        expected_mean = np.full(embedding_dim, 0.5)
        expected_log_std = np.full(embedding_dim, np.log(0.5))

        assert env.action_space.contains(latent)
        assert np.array_equal(latent, expected_embedding)
        assert np.array_equal(prob['mean'], expected_mean)
        assert np.array_equal(prob['log_std'], expected_log_std)

        latents, probs = embedding.get_latents([obs, obs, obs])
        for latent, mean, log_std in zip(latents, probs['mean'],
                                         probs['log_std']):
            assert env.action_space.contains(latent)
            assert np.array_equal(latent, expected_embedding)
            assert np.array_equal(mean, expected_mean)
            assert np.array_equal(log_std, expected_log_std)

    @pytest.mark.parametrize('obs_dim, embedding_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_dist_info_sym(self, obs_dim, embedding_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=embedding_dim))
        with mock.patch(('garage.tf.embeddings.'
                         'gaussian_mlp_embedding.GaussianMLPModel'),
                        new=SimpleGaussianMLPModel):
            embedding_spec = EmbeddingSpec(
                input_space=env.spec.observation_space,
                latent_space=env.spec.action_space)
            embedding = GaussianMLPEmbedding(embedding_spec)

        env.reset()
        obs, _, _, _ = env.step(1)

        obs_dim = env.spec.observation_space.flat_dim
        obs_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, obs_dim))

        dist1_sym = embedding.dist_info_sym(obs_ph, name='p1_sym')

        # flatten output
        expected_mean = [np.full(np.prod(embedding_dim), 0.5)]
        expected_log_std = [np.full(np.prod(embedding_dim), np.log(0.5))]

        prob = self.sess.run(dist1_sym, feed_dict={obs_ph: [obs.flatten()]})

        assert np.array_equal(prob['mean'], expected_mean)
        assert np.array_equal(prob['log_std'], expected_log_std)

    @pytest.mark.parametrize('obs_dim, embedding_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_is_pickleable(self, obs_dim, embedding_dim):
        env = TfEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=embedding_dim))
        with mock.patch(('garage.tf.embeddings.'
                         'gaussian_mlp_embedding.GaussianMLPModel'),
                        new=SimpleGaussianMLPModel):
            embedding_spec = EmbeddingSpec(
                input_space=env.spec.observation_space,
                latent_space=env.spec.action_space)
            embedding = GaussianMLPEmbedding(embedding_spec)

        env.reset()
        obs, _, _, _ = env.step(1)
        obs_dim = env.spec.observation_space.flat_dim

        with tf.compat.v1.variable_scope(
                'GaussianMLPEmbedding/GaussianMLPModel', reuse=True):
            return_var = tf.compat.v1.get_variable('return_var')
        # assign it to all one
        return_var.load(tf.ones_like(return_var).eval())
        output1 = self.sess.run(
            embedding.model.outputs[:-1],
            feed_dict={embedding.model.input: [obs.flatten()]})

        p = pickle.dumps(embedding)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            embedding_pickled = pickle.loads(p)
            output2 = sess.run(
                embedding_pickled.model.outputs[:-1],
                feed_dict={embedding_pickled.model.input: [obs.flatten()]})
            assert np.array_equal(output1, output2)

    def test_utils(self):
        first_space = akro.Box(low=np.array([-1, -1]), high=np.array([1, 1]))
        second_space = akro.Box(low=np.array([-2, -2]), high=np.array([2, 2]))
        concat_space = concat_spaces(first_space, second_space)
        low, high = concat_space.bounds
        np.testing.assert_equal(low, (-1, -1, -2, -2))
        np.testing.assert_equal(high, (1, 1, 2, 2))

    def test_auxiliary(self):
        input_space = akro.Box(np.array([-1, -1]), np.array([1, 1]))
        latent_space = akro.Box(np.array([-2, -2, -2]), np.array([2, 2, 2]))
        embedding_spec = EmbeddingSpec(input_space=input_space,
                                       latent_space=latent_space)
        embedding = GaussianMLPEmbedding(embedding_spec,
                                         hidden_sizes=[32, 32, 32])

        assert embedding.vectorized

        # 9 Layers: (3 hidden + 1 output) * (1 weight + 1 bias) + 1 log_std
        assert len(embedding.get_params()) == 9
        assert len(embedding.get_global_vars()) == 9

        assert embedding.distribution.dim == latent_space.shape[0]
        assert embedding.input.shape.as_list() == [None, input_space.shape[0]]
        assert (embedding.latent_mean.shape.as_list() == [
            None, latent_space.shape[0]
        ])
        assert (embedding.latent_std_param.shape.as_list() == [
            None, latent_space.shape[0]
        ])

        # To increase coverage in embeddings/base.py
        embedding.reset()
        assert embedding.embedding_spec == embedding_spec
        assert not embedding.recurrent

        assert embedding.state_info_specs == []
        assert embedding.state_info_keys == []

        var_shapes = [
            (2, 32),
            (32, ),  # input
            (32, 32),
            (32, ),  # hidden 0
            (32, 32),
            (32, ),  # hidden 1
            (32, 3),
            (3, ),  # hidden 2
            (3, )
        ]  # log_std
        assert sorted(embedding.get_param_shapes()) == sorted(var_shapes)

        var_count = sum(list(map(np.prod, var_shapes)))
        embedding.set_param_values(np.ones(var_count))
        assert (embedding.get_param_values() == np.ones(var_count)).all()

        assert (sorted(
            map(np.shape, embedding.flat_to_params(
                np.ones(var_count)))) == sorted(var_shapes))
