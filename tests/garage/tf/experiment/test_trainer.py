import pytest
import ray
import tensorflow as tf

from garage.envs import GymEnv
from garage.experiment import deterministic
from garage.np.baselines import LinearFeatureBaseline
from garage.plotter import Plotter
from garage.sampler import LocalSampler, MultiprocessingSampler, RaySampler
from garage.tf.algos import VPG
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase
from tests.fixtures.sampler import ray_session_fixture


class TestTrainer(TfGraphTestCase):

    def test_session(self):
        with TFTrainer(snapshot_config):
            assert tf.compat.v1.get_default_session() is not None, (
                'TFTrainer() should provide a default tf session.')

        sess = tf.compat.v1.Session()
        with TFTrainer(snapshot_config, sess=sess):
            assert tf.compat.v1.get_default_session() is sess, (
                'TFTrainer(sess) should use sess as default session.')

    def test_train(self):
        with TFTrainer(snapshot_config) as trainer:
            env = GymEnv('CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(8, 8))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       discount=0.99,
                       optimizer_args=dict(learning_rate=0.01, ))

            trainer.setup(algo, env)
            trainer.train(n_epochs=1, batch_size=100)

    def test_external_sess(self):
        with tf.compat.v1.Session() as sess:
            with TFTrainer(snapshot_config, sess=sess):
                pass
            # sess should still be the default session here.
            tf.no_op().run()

    def test_set_plot(self):
        deterministic.set_seed(1)
        with TFTrainer(snapshot_config) as trainer:
            env = GymEnv('CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(8, 8))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       discount=0.99,
                       optimizer_args=dict(learning_rate=0.01, ))

            trainer.setup(algo, env)
            trainer.train(n_epochs=1, batch_size=100, plot=True)

            assert isinstance(trainer._plotter, Plotter), (
                'self.plotter in TFTrainer should be set to Plotter.')

    def test_call_train_before_set_up(self):
        with pytest.raises(Exception):
            with TFTrainer(snapshot_config) as trainer:
                trainer.train(n_epochs=1, batch_size=100)

    def test_call_save_before_set_up(self):
        with pytest.raises(Exception):
            with TFTrainer(snapshot_config) as trainer:
                trainer.save(0)

    def test_make_sampler_local_sampler(self):
        with TFTrainer(snapshot_config) as trainer:
            env = GymEnv('CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(8, 8))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       discount=0.99,
                       optimizer_args=dict(learning_rate=0.01, ))

            trainer.setup(algo, env, sampler_cls=LocalSampler)
            assert isinstance(trainer._sampler, LocalSampler)
            trainer.train(n_epochs=1, batch_size=10)

    def test_make_sampler_ray_sampler(self, ray_session_fixture):
        del ray_session_fixture
        assert ray.is_initialized()
        with TFTrainer(snapshot_config) as trainer:
            env = GymEnv('CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(8, 8))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       discount=0.99,
                       optimizer_args=dict(learning_rate=0.01, ))

            trainer.setup(algo, env, sampler_cls=RaySampler)
            assert isinstance(trainer._sampler, RaySampler)
            trainer.train(n_epochs=1, batch_size=10)
