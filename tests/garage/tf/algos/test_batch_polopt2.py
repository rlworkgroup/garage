from unittest import mock

import numpy as np
import pytest

from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import BatchPolopt2
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalLSTMPolicy
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.policies import GaussianLSTMPolicy
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestBatchPolopt2(TfGraphTestCase):

    # This test cause low memory with some reason
    @pytest.mark.flaky
    @mock.patch.multiple(BatchPolopt2, __abstractmethods__=set())
    # pylint: disable=abstract-class-instantiated, no-member
    def test_process_samples_continuous_non_recurrent(self):
        env = TfEnv(DummyBoxEnv())
        policy = GaussianMLPPolicy(env_spec=env.spec)
        baseline = GaussianMLPBaseline(env_spec=env.spec)
        max_path_length = 100
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = BatchPolopt2(env_spec=env.spec,
                                policy=policy,
                                baseline=baseline,
                                max_path_length=max_path_length,
                                flatten_input=True)
            runner.setup(algo, env, sampler_args=dict(n_envs=1))
            runner.train(n_epochs=1, batch_size=max_path_length)
            paths = runner.obtain_samples(0)
            samples = algo.process_samples(0, paths)
            # Since there is only 1 vec_env in the sampler and DummyBoxEnv
            # never terminate until it reaches max_path_length, batch size
            # must be max_path_length, i.e. 100
            assert samples['observations'].shape == (
                max_path_length, env.observation_space.flat_dim)
            assert samples['actions'].shape == (max_path_length,
                                                env.action_space.flat_dim)
            assert samples['rewards'].shape == (max_path_length, )
            assert samples['baselines'].shape == (max_path_length, )
            assert samples['returns'].shape == (max_path_length, )
            # there is only 1 path
            assert samples['lengths'].shape == (1, )
            # non-recurrent policy has empty agent info
            assert samples['agent_infos'] == {}
            # DummyBoxEnv has env_info dummy
            assert samples['env_infos']['dummy'].shape == (max_path_length, )
            assert isinstance(samples['average_return'], float)

    # This test cause low memory with some reason
    @pytest.mark.flaky
    # pylint: disable=abstract-class-instantiated, no-member
    @mock.patch.multiple(BatchPolopt2, __abstractmethods__=set())
    def test_process_samples_continuous_recurrent(self):
        env = TfEnv(DummyBoxEnv())
        policy = GaussianLSTMPolicy(env_spec=env.spec)
        baseline = GaussianMLPBaseline(env_spec=env.spec)
        max_path_length = 100
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = BatchPolopt2(env_spec=env.spec,
                                policy=policy,
                                baseline=baseline,
                                max_path_length=max_path_length,
                                flatten_input=True)
            runner.setup(algo, env, sampler_args=dict(n_envs=1))
            runner.train(n_epochs=1, batch_size=max_path_length)
            paths = runner.obtain_samples(0)
            samples = algo.process_samples(0, paths)
            # Since there is only 1 vec_env in the sampler and DummyBoxEnv
            # never terminate until it reaches max_path_length, batch size
            # must be max_path_length, i.e. 100
            assert samples['observations'].shape == (
                max_path_length, env.observation_space.flat_dim)
            assert samples['actions'].shape == (max_path_length,
                                                env.action_space.flat_dim)
            assert samples['rewards'].shape == (max_path_length, )
            assert samples['baselines'].shape == (max_path_length, )
            assert samples['returns'].shape == (max_path_length, )
            # there is only 1 path
            assert samples['lengths'].shape == (1, )
            for key, shape in policy.state_info_specs:
                assert samples['agent_infos'][key].shape == (max_path_length,
                                                             np.prod(shape))
            # DummyBoxEnv has env_info dummy
            assert samples['env_infos']['dummy'].shape == (max_path_length, )
            assert isinstance(samples['average_return'], float)

    # This test cause low memory with some reason
    @pytest.mark.flaky
    # pylint: disable=abstract-class-instantiated, no-member
    @mock.patch.multiple(BatchPolopt2, __abstractmethods__=set())
    def test_process_samples_discrete_non_recurrent(self):
        env = TfEnv(DummyDiscreteEnv())
        policy = CategoricalMLPPolicy(env_spec=env.spec)
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        max_path_length = 100
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = BatchPolopt2(env_spec=env.spec,
                                policy=policy,
                                baseline=baseline,
                                max_path_length=max_path_length,
                                flatten_input=True)
            runner.setup(algo, env, sampler_args=dict(n_envs=1))
            runner.train(n_epochs=1, batch_size=max_path_length)
            paths = runner.obtain_samples(0)
            samples = algo.process_samples(0, paths)
            # Since there is only 1 vec_env in the sampler and DummyDiscreteEnv
            # always terminate, number of paths must be max_path_length, and
            # batch size must be max_path_length as well, i.e. 100
            assert samples['observations'].shape == (
                max_path_length, env.observation_space.flat_dim)
            assert samples['actions'].shape == (max_path_length,
                                                env.action_space.n)
            assert samples['rewards'].shape == (max_path_length, )
            assert samples['baselines'].shape == (max_path_length, )
            assert samples['returns'].shape == (max_path_length, )
            # there is 100 path
            assert samples['lengths'].shape == (max_path_length, )
            # non-recurrent policy has empty agent info
            assert samples['agent_infos'] == {}
            # non-recurrent policy has empty env info
            assert samples['env_infos'] == {}
            assert isinstance(samples['average_return'], float)

    # This test cause low memory with some reason
    @pytest.mark.flaky
    # pylint: disable=abstract-class-instantiated, no-member
    @mock.patch.multiple(BatchPolopt2, __abstractmethods__=set())
    def test_process_samples_discrete_recurrent(self):
        env = TfEnv(DummyDiscreteEnv())
        policy = CategoricalLSTMPolicy(env_spec=env.spec)
        baseline = LinearFeatureBaseline(env_spec=env.spec)
        max_path_length = 100
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            algo = BatchPolopt2(env_spec=env.spec,
                                policy=policy,
                                baseline=baseline,
                                max_path_length=max_path_length,
                                flatten_input=True)
            runner.setup(algo, env, sampler_args=dict(n_envs=1))
            runner.train(n_epochs=1, batch_size=max_path_length)
            paths = runner.obtain_samples(0)
            samples = algo.process_samples(0, paths)
            # Since there is only 1 vec_env in the sampler and DummyDiscreteEnv
            # always terminate, number of paths must be max_path_length, and
            # batch size must be max_path_length as well, i.e. 100
            assert samples['observations'].shape == (
                max_path_length, env.observation_space.flat_dim)
            assert samples['actions'].shape == (max_path_length,
                                                env.action_space.n)
            assert samples['rewards'].shape == (max_path_length, )
            assert samples['baselines'].shape == (max_path_length, )
            assert samples['returns'].shape == (max_path_length, )
            # there is 100 path
            assert samples['lengths'].shape == (max_path_length, )
            # non-recurrent policy has empty agent info
            for key, shape in policy.state_info_specs:
                assert samples['agent_infos'][key].shape == (max_path_length,
                                                             np.prod(shape))
            # non-recurrent policy has empty env info
            assert samples['env_infos'] == {}
            assert isinstance(samples['average_return'], float)
