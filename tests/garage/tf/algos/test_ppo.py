"""
This script creates a test that fails when garage.tf.algos.PPO performance is
too low.
"""
import gym
import tensorflow as tf

from garage.envs import normalize
import garage.misc.logger as logger
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianLSTMPolicy
from garage.tf.policies import GaussianMLPPolicy
from tests.fixtures import TfGraphTestCase


class TestPPO(TfGraphTestCase):
    def test_ppo_pendulum(self):
        """Test PPO with Pendulum environment."""
        logger.reset()
        env = TfEnv(normalize(gym.make("InvertedDoublePendulum-v2")))
        policy = GaussianMLPPolicy(
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
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=2048,
            max_path_length=100,
            n_itr=10,
            discount=0.99,
            lr_clip_range=0.01,
            optimizer_args=dict(batch_size=32, max_epochs=10),
            plot=False,
        )
        last_avg_ret = algo.train(sess=self.sess)
        assert last_avg_ret > 40

        env.close()

    def test_ppo_pendulum_recurrent(self):
        """Test PPO with Pendulum environment and recurrent policy."""
        logger.reset()
        env = TfEnv(normalize(gym.make("InvertedDoublePendulum-v2")))
        policy = GaussianLSTMPolicy(env_spec=env.spec, )
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(hidden_sizes=(32, 32)),
        )
        algo = PPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=2048,
            max_path_length=100,
            n_itr=10,
            discount=0.99,
            lr_clip_range=0.01,
            optimizer_args=dict(batch_size=32, max_epochs=10),
            plot=False,
        )
        last_avg_ret = algo.train(sess=self.sess)
        assert last_avg_ret > 40

        env.close()
