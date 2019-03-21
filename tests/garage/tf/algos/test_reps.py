"""
This script creates a test that fails when garage.tf.algos.REPS performance is
too low.
"""
import gym

from garage.baselines import LinearFeatureBaseline
from garage.experiment import LocalRunner
import garage.misc.logger as logger
from garage.tf.algos import REPS
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestREPS(TfGraphTestCase):
    def test_reps_cartpole(self):
        """Test REPS with gym Cartpole environment."""
        with LocalRunner(self.sess) as runner:
            logger.reset()
            env = TfEnv(gym.make("CartPole-v0"))

            policy = CategoricalMLPPolicy(
                env_spec=env.spec, hidden_sizes=[32, 32])

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = REPS(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                batch_size=4000,
                max_path_length=100,
                n_itr=10,
                discount=0.99,
                max_kl_step=1e6)

            runner.setup(algo, env)

            last_avg_ret = runner.train(n_epochs=10, batch_size=4000)
            assert last_avg_ret > 5

            env.close()
