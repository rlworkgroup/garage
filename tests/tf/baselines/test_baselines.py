"""
This script creates a test that fails when
garage.tf.baselines failed to initialize.
"""
import unittest

from tests.envs.dummy import DummyBoxEnv

from garage.tf.baselines import DeterministicMLPBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv


class TestTfBaselines(unittest.TestCase):
    def test_baseline(self):
        """Test the baseline initialization."""
        box_env = TfEnv(DummyBoxEnv())
        deterministic_mlp_baseline = DeterministicMLPBaseline(env_spec=box_env)
        gaussian_mlp_baseline = GaussianMLPBaseline(env_spec=box_env)
