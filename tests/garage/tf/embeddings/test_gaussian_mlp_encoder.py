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

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestGaussianMLPEncoder(TfGraphTestCase):

    @pytest.mark.parametrize('obs_dim, embedding_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_get_embedding(self, obs_dim, embedding_dim):
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=embedding_dim))
        embedding_spec = InOutSpec(input_space=env.spec.observation_space,
                                   output_space=env.spec.action_space)
        embedding = GaussianMLPEncoder(embedding_spec)
        task_input = tf.compat.v1.placeholder(tf.float32,
                                              shape=(None, None,
                                                     embedding.input_dim))
        embedding.build(task_input, name='task_input')

        env.reset()
        obs = env.step(env.action_space.sample()).observation

        latent, _ = embedding.get_latent(obs)
        latents, _ = embedding.get_latents([obs] * 5)
        assert env.action_space.contains(latent)
        for latent in latents:
            assert env.action_space.contains(latent)

    @pytest.mark.parametrize('obs_dim, embedding_dim', [
        ((1, ), (1, )),
        ((1, ), (2, )),
        ((2, ), (2, )),
        ((1, 1), (1, 1)),
        ((1, 1), (2, 2)),
        ((2, 2), (2, 2)),
    ])
    def test_is_pickleable(self, obs_dim, embedding_dim):
        env = GymEnv(DummyBoxEnv(obs_dim=obs_dim, action_dim=embedding_dim))
        embedding_spec = InOutSpec(input_space=env.spec.observation_space,
                                   output_space=env.spec.action_space)
        embedding = GaussianMLPEncoder(embedding_spec)

        env.reset()
        obs = env.step(env.action_space.sample()).observation
        obs_dim = env.spec.observation_space.flat_dim

        with tf.compat.v1.variable_scope('GaussianMLPEncoder/GaussianMLPModel',
                                         reuse=True):
            bias = tf.compat.v1.get_variable(
                'dist_params/mean_network/hidden_0/bias')
        # assign it to all one
        bias.load(tf.ones_like(bias).eval())
        output1 = self.sess.run(
            [embedding.distribution.loc,
             embedding.distribution.stddev()],
            feed_dict={embedding.model.input: [[obs.flatten()]]})

        p = pickle.dumps(embedding)
        with tf.compat.v1.Session(graph=tf.Graph()) as sess:
            embedding_pickled = pickle.loads(p)

            output2 = sess.run(
                [
                    embedding_pickled.distribution.loc,
                    embedding_pickled.distribution.stddev()
                ],
                feed_dict={embedding_pickled.model.input: [[obs.flatten()]]})
            assert np.array_equal(output1, output2)

    def test_clone(self):
        env = GymEnv(DummyBoxEnv(obs_dim=(2, ), action_dim=(2, )))
        embedding_spec = InOutSpec(input_space=env.spec.observation_space,
                                   output_space=env.spec.action_space)
        embedding = GaussianMLPEncoder(embedding_spec)
        clone_embedding = embedding.clone(name='cloned')
        assert clone_embedding.input_dim == embedding.input_dim
        for cloned_param, param in zip(
                clone_embedding.model.parameters.values(),
                embedding.model.parameters.values()):
            assert np.array_equal(cloned_param, param)
        assert clone_embedding.output_dim == embedding.output_dim

    def test_auxiliary(self):
        input_space = akro.Box(np.array([-1, -1]), np.array([1, 1]))
        latent_space = akro.Box(np.array([-2, -2, -2]), np.array([2, 2, 2]))
        embedding_spec = InOutSpec(input_space=input_space,
                                   output_space=latent_space)
        embedding = GaussianMLPEncoder(embedding_spec,
                                       hidden_sizes=[32, 32, 32])
        # 9 Layers: (3 hidden + 1 output) * (1 weight + 1 bias) + 1 log_std
        assert len(embedding.get_params()) == 9
        assert len(embedding.get_global_vars()) == 9

        assert embedding.distribution.loc.get_shape().as_list(
        )[-1] == latent_space.shape[0]
        assert embedding.input.shape.as_list() == [
            None, None, input_space.shape[0]
        ]
        assert (embedding.latent_mean.shape.as_list() == [
            None, None, latent_space.shape[0]
        ])
        assert (embedding.latent_std_param.shape.as_list() == [
            None, 1, latent_space.shape[0]
        ])

        # To increase coverage in embeddings/base.py
        embedding.reset()
        assert embedding.input_dim == embedding_spec.input_space.flat_dim
        assert embedding.output_dim == embedding_spec.output_space.flat_dim

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
