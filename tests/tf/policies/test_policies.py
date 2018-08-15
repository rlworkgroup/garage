"""
This script creates a test that fails when
garage.tf.policies failed to initialize.
"""

import unittest

from tests.envs.dummy import DummyBoxEnv, DummyDiscreteEnv

from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.policies import DeterministicMLPPolicy
from garage.tf.policies import GaussianGRUPolicy
from garage.tf.policies import GaussianLSTMPolicy
from garage.tf.policies import GaussianMLPPolicy


class TestTfPolicies(unittest.TestCase):
    def test_policies(self):
        """Test the policies initialization."""
        box_env = TfEnv(DummyBoxEnv())
        discrete_env = TfEnv(DummyDiscreteEnv())
        categorical_mlp_policy = CategoricalMLPPolicy(
            env_spec=discrete_env, hidden_sizes=(1, ))
        continuous_mlp_policy = ContinuousMLPPolicy(
            env_spec=box_env, hidden_sizes=(1, ))
        deterministic_mlp_policy = DeterministicMLPPolicy(
            env_spec=box_env, hidden_sizes=(1, ))
        gaussian_gru_policy = GaussianGRUPolicy(env_spec=box_env, hidden_dim=1)
        gaussian_lstm_policy = GaussianLSTMPolicy(
            env_spec=box_env, hidden_dim=1)
        gaussian_mlp_policy = GaussianMLPPolicy(
            env_spec=box_env, hidden_sizes=(1, ))
