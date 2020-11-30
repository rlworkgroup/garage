"""
This script creates a unittest that tests Categorical policies in
garage.tf.policies.
"""
# yapf: disable
import pytest

from garage.envs import GymEnv, normalize
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TRPO
from garage.tf.optimizers import (ConjugateGradientOptimizer,
                                  FiniteDifferenceHVP)
from garage.tf.policies import (CategoricalGRUPolicy, CategoricalLSTMPolicy,
                                CategoricalMLPPolicy)
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase

# yapf: enable

policies = [CategoricalGRUPolicy, CategoricalLSTMPolicy, CategoricalMLPPolicy]


class TestCategoricalPolicies(TfGraphTestCase):

    @pytest.mark.parametrize('policy_cls', [*policies])
    def test_categorical_policies(self, policy_cls):
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            env = normalize(GymEnv('CartPole-v0', max_episode_length=100))

            policy = policy_cls(name='policy', env_spec=env.spec)

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            sampler = LocalSampler(
                agents=policy,
                envs=env,
                max_episode_length=env.spec.max_episode_length,
                is_tf_worker=True)

            algo = TRPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                sampler=sampler,
                discount=0.99,
                max_kl_step=0.01,
                optimizer=ConjugateGradientOptimizer,
                optimizer_args=dict(hvp_approach=FiniteDifferenceHVP(
                    base_eps=1e-5)),
            )

            trainer.setup(algo, env)
            trainer.train(n_epochs=1, batch_size=4000)

            env.close()
