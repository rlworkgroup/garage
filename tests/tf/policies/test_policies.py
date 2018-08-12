"""
This script creates a test that fails when 
garage.tf.policies failed to initialize.
"""

import unittest

import gym
import numpy as np

from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.policies import GaussianGRUPolicy
from garage.tf.policies import GaussianLSTMPolicy
from garage.tf.policies import GaussianMLPPolicy


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


class DummyDiscreteEnv(gym.Env):
    """A dummy box environment."""

    @property
    def observation_space(self):
        """Return a observation space."""
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

    @property
    def action_space(self):
        """Return a action space."""
        return gym.spaces.Discrete(2)

    def reset(self):
        """Reset the environment."""
        return np.zeros(1)

    def step(self, action):
        """Step the environment."""
        return np.zeros(1), 0, True, dict()


class TestTfPolicies(unittest.TestCase):
    def test_policies(self):
        """Test the policies initialization."""
        box_env = TfEnv(DummyBoxEnv())
        discrete_env = TfEnv(DummyDiscreteEnv())
        categorical_mlp_policy = CategoricalMLPPolicy(
            env_spec=discrete_env, hidden_sizes=(1, ))
        continuous_mlp_policy = ContinuousMLPPolicy(
            env_spec=box_env, hidden_sizes=(1, ))
        gaussian_gru_policy = GaussianGRUPolicy(env_spec=box_env, hidden_dim=1)
        gaussian_lstm_policy = GaussianLSTMPolicy(
            env_spec=box_env, hidden_dim=1)
        gaussian_mlp_policy = GaussianMLPPolicy(
            env_spec=box_env, hidden_sizes=(1, ))
