import gym
import pytest

from garage.envs import normalize
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import GaussianGRUPolicy
from garage.tf.policies import GaussianLSTMPolicy
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase

policies = [GaussianGRUPolicy, GaussianLSTMPolicy, GaussianMLPPolicy]


class TestGaussianPolicies(TfGraphTestCase):

    @pytest.mark.parametrize('policy_cls', policies)
    def test_gaussian_policies(self, policy_cls):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = TfEnv(normalize(gym.make('Pendulum-v0')))

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

            runner.setup(algo, env)
            runner.train(n_epochs=1, batch_size=4000)
            env.close()
