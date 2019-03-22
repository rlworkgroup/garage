"""
This script creates a test that fails when garage.tf.algos.PPO performance is
too low.
"""
import gym
import tensorflow as tf

from garage.envs import normalize
from garage.experiment import LocalRunner
import garage.misc.logger as logger
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicyWithModel
from tests.fixtures import TfGraphTestCase


class TestPPO2(TfGraphTestCase):
    def test_ppo_pendulum_with_model(self):
        """Test PPO with model, with Pendulum environment."""
        with LocalRunner(self.sess) as runner:
            logger.reset()
            env = TfEnv(normalize(gym.make("InvertedDoublePendulum-v2")))
            policy = GaussianMLPPolicyWithModel(
                env_spec=env.spec,
                hidden_sizes=(64, 64),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=None,
            )
            baseline = GaussianMLPBaseline(
                env_spec=env.spec,
                regressor_args=dict(hidden_sizes=(32, 32)),
            )
            algo = PPO(
                env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=100,
                discount=0.99,
                lr_clip_range=0.01,
                optimizer_args=dict(batch_size=32, max_epochs=10),
            )
            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=10, batch_size=2048)
            assert last_avg_ret > 40

            env.close()
