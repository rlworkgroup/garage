import unittest

import tensorflow as tf

from garage.baselines import LinearFeatureBaseline
from garage.experiment import LocalRunner
from garage.sampler import singleton_pool
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.samplers import BatchSampler


class TestLocalRunner(unittest.TestCase):
    def tearDown(self):
        tf.reset_default_graph()

    def test_session(self):
        with LocalRunner():
            self.assertIsNotNone(
                tf.get_default_session(),
                "LocalRunner() should provide a default tf session.")

        sess = tf.Session()
        with LocalRunner(sess=sess):
            self.assertIs(
                tf.get_default_session(), sess,
                "LocalRunner(sess) should use sess as default session.")

    def test_singleton_pool(self):
        max_cpus = 8
        with LocalRunner(max_cpus=max_cpus):
            self.assertEqual(
                max_cpus, singleton_pool.n_parallel,
                "LocaRunner(max_cpu) should set up singleton_pool.")

    def test_batch_sampler(self):
        max_cpus = 8
        with LocalRunner(max_cpus=max_cpus) as runner:
            env = TfEnv(env_name='CartPole-v1')

            policy = CategoricalMLPPolicy(
                name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=1,
                whole_paths=True,
                discount=0.99)

            runner.setup(
                algo,
                env,
                sampler_cls=BatchSampler,
                sampler_args={'n_envs': max_cpus})

            try:
                runner.initialize_tf_vars()
            except BaseException:
                raise self.failureException(
                    "LocalRunner should be able to initialize tf variables.")

            runner.start_worker()

            paths = runner.sampler.obtain_samples(0, 8)
            self.assertGreaterEqual(
                len(paths), max_cpus, "BatchSampler should sample more than "
                "max_cpus=%d trajectories" % max_cpus)

    # Note:
    #   test_batch_sampler should pass if tested independently
    #   from other tests, but cannot be tested on CI.
    #
    #   This is because nose2 runs all tests in a single process,
    #   when this test is run, tensorflow has already been initialized, and
    #   later singleton_pool will hangs because tensorflow is not fork-safe.
    test_batch_sampler.flaky = True

    def test_train(self):
        with LocalRunner() as runner:
            env = TfEnv(env_name='CartPole-v1')

            policy = CategoricalMLPPolicy(
                name="policy", env_spec=env.spec, hidden_sizes=(8, 8))

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
