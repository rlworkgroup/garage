"""
This script creates a unittest that tests Categorical policies in
garage.tf.policies.
"""
# yapf: disable
import pytest

from garage.envs import GymEnv, normalize
from garage.experiment import LocalTFRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.optimizers import (ConjugateGradientOptimizer,
                                  FiniteDifferenceHvp)
from garage.tf.policies import (CategoricalGRUPolicy,
                                CategoricalLSTMPolicy,
                                CategoricalMLPPolicy)

from tests.fixtures import snapshot_config, TfGraphTestCase

# yapf: enable

policies = [CategoricalGRUPolicy, CategoricalLSTMPolicy, CategoricalMLPPolicy]


class TestCategoricalPolicies(TfGraphTestCase):

    @pytest.mark.parametrize('policy_cls', [*policies])
    def test_categorical_policies(self, policy_cls):
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = normalize(GymEnv('CartPole-v0'))

            policy = policy_cls(name='policy', env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_episode_length=100,
                discount=0.99,
                max_kl_step=0.01,
                optimizer=ConjugateGradientOptimizer,
                optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                    base_eps=1e-5)),
            )

            runner.setup(algo, env, sampler_cls=LocalSampler)
            runner.train(n_epochs=1, batch_size=4000)

            env.close()
