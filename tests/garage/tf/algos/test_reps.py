"""
This script creates a test that fails when garage.tf.algos.REPS performance is
too low.
"""
import pytest

from garage.envs import GymEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import REPS
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer

from tests.fixtures import snapshot_config, TfGraphTestCase


class TestREPS(TfGraphTestCase):

    @pytest.mark.large
    def test_reps_cartpole(self):
        """Test REPS with gym Cartpole environment."""
        with TFTrainer(snapshot_config, sess=self.sess) as trainer:
            env = GymEnv('CartPole-v0')

            policy = CategoricalMLPPolicy(env_spec=env.spec,
                                          hidden_sizes=[32, 32])

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = REPS(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        discount=0.99)

            trainer.setup(algo, env, sampler_cls=LocalSampler)

            last_avg_ret = trainer.train(n_epochs=10, batch_size=4000)
            assert last_avg_ret > 5

            env.close()
