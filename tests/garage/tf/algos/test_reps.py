"""
This script creates a test that fails when garage.tf.algos.REPS performance is
too low.
"""
import gym
import pytest

from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import REPS
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestREPS(TfGraphTestCase):

    @pytest.mark.large
    def test_reps_cartpole(self):
        """Test REPS with gym Cartpole environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
            env = TfEnv(gym.make('CartPole-v0'))

            policy = CategoricalMLPPolicy(env_spec=env.spec,
                                          hidden_sizes=[32, 32])

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = REPS(env_spec=env.spec,
                        policy=policy,
                        baseline=baseline,
                        max_path_length=100,
                        discount=0.99)

            runner.setup(algo, env)

            last_avg_ret = runner.train(n_epochs=10, batch_size=4000)
            assert last_avg_ret > 5

            env.close()
