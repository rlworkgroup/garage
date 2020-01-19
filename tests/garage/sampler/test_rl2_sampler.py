import numpy as np
import pytest

from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RL2Sampler
from garage.tf.algos import PPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestRL2Sampler(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        self.meta_batch_size = 10
        self.max_path_length = 100
        self.env = TfEnv(HalfCheetahVelEnv())

    def test_rl2_sampler_n_envs(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            policy = GaussianMLPPolicy(env_spec=self.env.spec,
                                       hidden_sizes=[32, 32])

            baseline = LinearFeatureBaseline(env_spec=self.env.spec)

            algo = PPO(env_spec=self.env.spec,
                       policy=policy,
                       baseline=baseline,
                       max_path_length=self.max_path_length,
                       discount=0.99)
            runner.setup(algo,
                         env=self.env,
                         sampler_cls=RL2Sampler,
                         sampler_args=dict(
                             meta_batch_size=self.meta_batch_size,
                             n_envs=self.meta_batch_size))
            runner._start_worker()
            assert isinstance(runner._sampler, RL2Sampler)
            assert runner._sampler._envs_per_worker == 1
            assert all(runner._sampler._vec_envs_indices[0] == np.arange(
                self.meta_batch_size))
            paths = runner._sampler.obtain_samples(0)
            assert len(paths) == self.meta_batch_size
            assert len(paths[0]['observations']) == self.max_path_length
            paths = runner._sampler.obtain_samples(
                0, self.meta_batch_size * 10 * self.max_path_length)
            assert len(paths) == self.meta_batch_size * 10
            assert len(paths[0]['observations']) == self.max_path_length

    def test_rl2_sampler_more_envs_than_meta_batch(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            policy = GaussianMLPPolicy(env_spec=self.env.spec,
                                       hidden_sizes=[32, 32])

            baseline = LinearFeatureBaseline(env_spec=self.env.spec)

            algo = PPO(env_spec=self.env.spec,
                       policy=policy,
                       baseline=baseline,
                       max_path_length=self.max_path_length,
                       discount=0.99)
            runner.setup(algo,
                         env=self.env,
                         sampler_cls=RL2Sampler,
                         sampler_args=dict(
                             meta_batch_size=self.meta_batch_size,
                             n_envs=self.meta_batch_size * 2))
            runner._start_worker()
            assert isinstance(runner._sampler, RL2Sampler)
            assert runner._sampler._envs_per_worker == 2
            assert all(runner._sampler._vec_envs_indices[0] == np.arange(
                self.meta_batch_size))
            paths = runner._sampler.obtain_samples(0, whole_paths=True)
            # whole paths
            assert len(paths) == self.meta_batch_size * 2
            assert len(paths[0]['observations']) == self.max_path_length
            paths = runner._sampler.obtain_samples(
                0, self.meta_batch_size * 10 * self.max_path_length)
            assert len(paths) == self.meta_batch_size * 10
            assert len(paths[0]['observations']) == self.max_path_length

    def test_rl2_sampler_less_envs_than_meta_batch(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            policy = GaussianMLPPolicy(env_spec=self.env.spec,
                                       hidden_sizes=[32, 32])

            baseline = LinearFeatureBaseline(env_spec=self.env.spec)

            algo = PPO(env_spec=self.env.spec,
                       policy=policy,
                       baseline=baseline,
                       max_path_length=self.max_path_length,
                       discount=0.99)
            runner.setup(algo,
                         env=self.env,
                         sampler_cls=RL2Sampler,
                         sampler_args=dict(
                             meta_batch_size=self.meta_batch_size,
                             n_envs=self.meta_batch_size // 2))
            runner._start_worker()
            assert isinstance(runner._sampler, RL2Sampler)
            assert runner._sampler._envs_per_worker == 1
            all_indices = np.arange(self.meta_batch_size)
            for i in range(self.meta_batch_size // 2):
                assert all(runner._sampler._vec_envs_indices[i] ==
                           all_indices[i * 2:i * 2 + 2])
            paths = runner._sampler.obtain_samples(0)
            assert len(paths) == self.meta_batch_size
            assert len(paths[0]['observations']) == self.max_path_length
            paths = runner._sampler.obtain_samples(
                0, self.meta_batch_size * 10 * self.max_path_length)
            assert len(paths) == self.meta_batch_size * 10
            assert len(paths[0]['observations']) == self.max_path_length

    def test_rl2_sampler_invalid_num_of_env(self):
        with pytest.raises(
                ValueError,
                match='meta_batch_size must be a multiple of n_envs'):
            with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
                policy = GaussianMLPPolicy(env_spec=self.env.spec,
                                           hidden_sizes=[32, 32])

                baseline = LinearFeatureBaseline(env_spec=self.env.spec)

                algo = PPO(env_spec=self.env.spec,
                           policy=policy,
                           baseline=baseline,
                           max_path_length=self.max_path_length,
                           discount=0.99)
                runner.setup(algo,
                             env=self.env,
                             sampler_cls=RL2Sampler,
                             sampler_args=dict(
                                 meta_batch_size=self.meta_batch_size,
                                 n_envs=self.meta_batch_size - 1))
                runner._start_worker()
                runner._sampler.obtain_samples(0)

    def test_rl2_sampler_invalid_num_of_env_again(self):
        with pytest.raises(
                ValueError,
                match='n_envs must be a multiple of meta_batch_size'):
            with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
                policy = GaussianMLPPolicy(env_spec=self.env.spec,
                                           hidden_sizes=[32, 32])

                baseline = LinearFeatureBaseline(env_spec=self.env.spec)

                algo = PPO(env_spec=self.env.spec,
                           policy=policy,
                           baseline=baseline,
                           max_path_length=self.max_path_length,
                           discount=0.99)
                runner.setup(algo,
                             env=self.env,
                             sampler_cls=RL2Sampler,
                             sampler_args=dict(
                                 meta_batch_size=self.meta_batch_size,
                                 n_envs=self.meta_batch_size + 1))
                runner._start_worker()
                runner._sampler.obtain_samples(0)
