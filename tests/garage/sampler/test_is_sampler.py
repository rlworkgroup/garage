import unittest.mock

import gym

from garage.envs import normalize
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import ISSampler
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestISSampler(TfGraphTestCase):

    def test_is_sampler(self):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = TfEnv(normalize(gym.make('InvertedPendulum-v2')))
            policy = GaussianMLPPolicy(env_spec=env.spec,
                                       hidden_sizes=(32, 32))
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = TRPO(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_path_length=100,
                        discount=0.99,
                        max_kl_step=0.01)

            runner.setup(algo,
                         env,
                         sampler_cls=ISSampler,
                         sampler_args=dict(n_backtrack=1, init_is=1))
            runner._start_worker()

            paths = runner.sampler.obtain_samples(1)
            assert paths == [], 'Should return empty paths if no history'

            # test importance and live sampling get called alternatively
            with unittest.mock.patch.object(ISSampler,
                                            '_obtain_is_samples') as mocked:
                assert runner.sampler.obtain_samples(2, 20)
                mocked.assert_not_called()

                assert runner.sampler.obtain_samples(3)
                mocked.assert_called_once_with(3, None, True)

            # test importance sampling for first n_is_pretrain iterations
            with unittest.mock.patch.object(ISSampler,
                                            '_obtain_is_samples') as mocked:
                runner.sampler.n_is_pretrain = 5
                runner.sampler.n_backtrack = 'all'
                runner.sampler.obtain_samples(4)

                mocked.assert_called_once_with(4, None, True)

            runner.sampler.obtain_samples(5)

            # test random draw important samples
            runner.sampler.randomize_draw = True
            assert runner.sampler.obtain_samples(6, 1)
            runner.sampler.randomize_draw = False

            runner.sampler.obtain_samples(7, 30)

            # test ess_threshold use
            runner.sampler.ess_threshold = 500
            paths = runner.sampler.obtain_samples(8, 30)
            assert paths == [], (
                'Should return empty paths when ess_threshold is large')
            runner.sampler.ess_threshold = 0

            # test random sample selection when len(paths) > batch size
            runner.sampler.n_is_pretrain = 15
            runner.sampler.obtain_samples(9, 10)
            runner.sampler.obtain_samples(10, 1)

            runner._shutdown_worker()
