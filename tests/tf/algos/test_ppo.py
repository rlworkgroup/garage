"""
This script creates a test that fails when garage.tf.algos.PPO performance is
too low.
"""
import unittest

import gym
import tensorflow as tf

from garage.envs import normalize
from garage.tf.algos import PPO
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy


class TestTRPO(unittest.TestCase):
    def test_trpo_pendulum(self):
        """Test PPO with Pendulum environment."""
        env = TfEnv(normalize(gym.make("Pendulum-v0")))
        policy = GaussianMLPPolicy(
            name="policy",
            env_spec=env.spec,
            hidden_sizes=(32, 32),
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
            batch_size=1024,
            max_path_length=100,
            n_itr=10,
            discount=0.99,
            gae_lambda=0.98,
            policy_ent_coeff=0.0,
            plot=False,
        )
        last_avg_ret = algo.train()
        assert last_avg_ret > -400
