from dowel import logger
import pytest

from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.samplers import BatchSampler
from tests.fixtures import snapshot_config
from tests.fixtures.logger import NullOutput


class TestSampler:

    def setup_method(self):
        logger.add_output(NullOutput())

    def teardown_method(self):
        logger.remove_all()

    # Note:
    #   test_batch_sampler should pass if tested independently
    #   from other tests, but cannot be tested on CI.
    #
    #   This is because nose2 runs all tests in a single process,
    #   when this test is running, tensorflow has already been initialized,
    #   and later singleton_pool will hang because tensorflow is not fork-safe.
    @pytest.mark.flaky
    def test_tf_batch_sampler(self):
        max_cpus = 8
        with LocalTFRunner(snapshot_config, max_cpus=max_cpus) as runner:
            env = TfEnv(env_name='CartPole-v1')

            policy = CategoricalMLPPolicy(name='policy',
                                          env_spec=env.spec,
                                          hidden_sizes=(32, 32))

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = VPG(env_spec=env.spec,
                       policy=policy,
                       baseline=baseline,
                       max_path_length=1,
                       discount=0.99)

            runner.setup(algo,
                         env,
                         sampler_cls=BatchSampler,
                         sampler_args={'n_envs': max_cpus})

            try:
                runner.initialize_tf_vars()
            except BaseException:
                raise AssertionError(
                    'LocalRunner should be able to initialize tf variables.')

            runner._start_worker()

            paths = runner.sampler.obtain_samples(0,
                                                  batch_size=8,
                                                  whole_paths=True)
            assert len(paths) >= max_cpus, (
                'BatchSampler should sample more than max_cpus={} '
                'trajectories'.format(max_cpus))
