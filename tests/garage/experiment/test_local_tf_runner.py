import pytest
import tensorflow as tf

from garage.experiment import LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import singleton_pool
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.plotter import Plotter
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestLocalRunner(TfGraphTestCase):
    def test_session(self):
        with LocalRunner():
            assert tf.get_default_session is not None, (
                'LocalRunner() should provide a default tf session.')

        sess = tf.Session()
        with LocalRunner(sess=sess):
            assert tf.get_default_session() is sess, (
                'LocalRunner(sess) should use sess as default session.')

    def test_singleton_pool(self):
        max_cpus = 8
        with LocalRunner(max_cpus=max_cpus):
            assert max_cpus == singleton_pool.n_parallel, (
                'LocalRunner(max_cpu) should set up singleton_pool.')

    def test_train(self):
        with LocalRunner() as runner:
            env = TfEnv(env_name='CartPole-v1')

            policy = CategoricalMLPPolicy(
                name='policy', env_spec=env.spec, hidden_sizes=(8, 8))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                optimizer_args=dict(
                    tf_optimizer_args=dict(learning_rate=0.01, )))

            runner.setup(algo, env)
            runner.train(n_epochs=1, batch_size=100)

    def test_external_sess(self):
        with tf.Session() as sess:
            with LocalRunner(sess=sess):
                pass
            # sess should still be the default session here.
            tf.no_op().run()

    def test_set_plot(self):
        with LocalRunner() as runner:
            env = TfEnv(env_name='CartPole-v1')

            policy = CategoricalMLPPolicy(
                name='policy', env_spec=env.spec, hidden_sizes=(8, 8))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                optimizer_args=dict(
                    tf_optimizer_args=dict(learning_rate=0.01, )))

            runner.setup(algo, env)
            runner.train(n_epochs=1, batch_size=100, plot=True)

            assert isinstance(runner.plotter, Plotter), (
                'self.plotter in LocalRunner should be set to Plotter.')

    def test_call_train_before_set_up(self):
        with pytest.raises(Exception):
            with LocalRunner() as runner:
                runner.train(n_epochs=1, batch_size=100)

    def test_call_save_before_set_up(self):
        with pytest.raises(Exception):
            with LocalRunner() as runner:
                runner.save(0)
