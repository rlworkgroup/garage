# yapf: disable
import gym
import pytest

from garage.envs import GarageEnv, normalize
from garage.experiment import LocalTFRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.optimizers import (ConjugateGradientOptimizer,
                                  FiniteDifferenceHvp)
from garage.tf.policies import (GaussianGRUPolicy,
                                GaussianLSTMPolicy,
                                GaussianMLPPolicy)

from tests.fixtures import snapshot_config, TfGraphTestCase

# yapf: enable

policies = [GaussianGRUPolicy, GaussianLSTMPolicy, GaussianMLPPolicy]


class TestGaussianPolicies(TfGraphTestCase):

    @pytest.mark.parametrize('policy_cls', policies)
    def test_gaussian_policies(self, policy_cls):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = GarageEnv(normalize(gym.make('Pendulum-v0')))

            policy = policy_cls(name='policy', env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                max_kl_step=0.01,
                optimizer=ConjugateGradientOptimizer,
                optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                    base_eps=1e-5)),
            )

            runner.setup(algo, env, sampler_cls=LocalSampler)
            runner.train(n_epochs=1, batch_size=4000)
            env.close()
