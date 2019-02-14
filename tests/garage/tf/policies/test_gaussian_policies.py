import gym
from nose2 import tools

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
import garage.misc.logger as logger
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import GaussianGRUPolicy
from garage.tf.policies import GaussianLSTMPolicy
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase

policies = [GaussianGRUPolicy, GaussianLSTMPolicy, GaussianMLPPolicy]


class TestGaussianPolicies(TfGraphTestCase):
    @tools.params(*policies)
    def test_gaussian_policies(self, policy_cls):
        logger.reset()
        env = TfEnv(normalize(gym.make("Pendulum-v0")))

        policy = policy_cls(name="policy", env_spec=env.spec)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=100,
            n_itr=1,
            discount=0.99,
            step_size=0.01,
            plot=True,
            optimizer=ConjugateGradientOptimizer,
            optimizer_args=dict(
                hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
        )
        algo.train(sess=self.sess)
        env.close()
