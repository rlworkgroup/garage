"""
This script creates a test that fails when 
garage.tf.baselines failed to initialize.
"""
import unittest

import gym
import numpy as np

from garage.tf.baselines import DeterministicMLPBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv


class DummyBoxEnv(gym.Env):
    """A dummy box environment."""

    @property
    def observation_space(self):
        """Return a observation space."""
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

    @property
    def action_space(self):
        """Return a action space."""
        return gym.spaces.Box(
            low=-5.0, high=5.0, shape=(1, ), dtype=np.float32)

    def reset(self):
        """Reset the environment."""
        return np.zeros(1)

    def step(self, action):
        """Step the environment."""
        return np.zeros(1), 0, True, dict()


class TestTfBaselines(unittest.TestCase):
    def test_baseline(self):
        """Test the baseline initialization."""
        box_env = TfEnv(DummyBoxEnv())
        deterministic_mlp_baseline = DeterministicMLPBaseline(env_spec=box_env)
        gaussian_mlp_baseline = GaussianMLPBaseline(env_spec=box_env)
