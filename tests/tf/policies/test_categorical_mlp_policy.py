"""
This script creates a unittest that tests CategoricalMLPPolicy in
garage.tf.policies.
"""
import unittest

import gym

from garage.baselines import LinearFeatureBaseline
from garage.envs import normalize
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy


class TestCategoricalMLPPolicy(unittest.TestCase):
    def test_categorical_mlp_policy(self):
        env = TfEnv(normalize(gym.make("CartPole-v0")))

        policy = CategoricalMLPPolicy(
            name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

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
            plot=True)
        algo.train()
