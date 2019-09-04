import pytest
import tensorflow as tf

from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import singleton_pool
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.plotter import Plotter
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestLocalRunner(TfGraphTestCase):

    def test_session(self):
        with LocalTFRunner(snapshot_config):
            assert tf.compat.v1.get_default_session() is not None, (
                'LocalTFRunner() should provide a default tf session.')

        sess = tf.compat.v1.Session()
        with LocalTFRunner(snapshot_config, sess=sess):
            assert tf.compat.v1.get_default_session() is sess, (
                'LocalTFRunner(sess) should use sess as default session.')

    def test_singleton_pool(self):
        max_cpus = 8
        with LocalTFRunner(snapshot_config, max_cpus=max_cpus):
            assert max_cpus == singleton_pool.n_parallel, (
                'LocalTFRunner(max_cpu) should set up singleton_pool.')

    def test_train(self):
        with LocalTFRunner(snapshot_config) as runner:
            env = TfEnv(env_name='CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(8, 8))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       max_path_length=100,
                       discount=0.99,
                       optimizer_args=dict(
                           tf_optimizer_args=dict(learning_rate=0.01, )))

            runner.setup(algo, env)
            runner.train(n_epochs=1, batch_size=100)

    def test_external_sess(self):
        with tf.compat.v1.Session() as sess:
            with LocalTFRunner(snapshot_config, sess=sess):
                pass
            # sess should still be the default session here.
            tf.no_op().run()

    def test_set_plot(self):
        with LocalTFRunner(snapshot_config) as runner:
            env = TfEnv(env_name='CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(8, 8))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       max_path_length=100,
                       discount=0.99,
                       optimizer_args=dict(
                           tf_optimizer_args=dict(learning_rate=0.01, )))

            runner.setup(algo, env)
            runner.train(n_epochs=1, batch_size=100, plot=True)

            assert isinstance(runner.plotter, Plotter), (
                'self.plotter in LocalTFRunner should be set to Plotter.')

    def test_call_train_before_set_up(self):
        with pytest.raises(Exception):
            with LocalTFRunner(snapshot_config) as runner:
                runner.train(n_epochs=1, batch_size=100)

    def test_call_save_before_set_up(self):
        with pytest.raises(Exception):
            with LocalTFRunner(snapshot_config) as runner:
                runner.save(0)
