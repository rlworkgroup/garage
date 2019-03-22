"""
This script creates a unittest that tests Categorical policies in
garage.tf.policies.
"""
import gym
from nose2 import tools

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.experiment import LocalRunner
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import CategoricalGRUPolicy
from garage.tf.policies import CategoricalLSTMPolicy
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import TfGraphTestCase

policies = [CategoricalGRUPolicy, CategoricalLSTMPolicy, CategoricalMLPPolicy]


class TestCategoricalPolicies(TfGraphTestCase):
    @tools.params(*policies)
    def test_categorical_policies(self, policy_cls):
        with LocalRunner(self.sess) as runner:
            env = TfEnv(normalize(gym.make("CartPole-v0")))

            policy = policy_cls(name="policy", env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                step_size=0.01,
                plot=True,
                optimizer=ConjugateGradientOptimizer,
                optimizer_args=dict(
                    hvp_approach=FiniteDifferenceHvp(base_eps=1e-5)),
            )

            runner.setup(algo, env)
            runner.train(n_epochs=1, batch_size=4000)

            env.close()
