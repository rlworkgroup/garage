"""
This script creates a test that fails when
garage.tf.baselines failed to initialize.
"""
from garage.tf.baselines import DeterministicMLPBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestTfBaselines(TfGraphTestCase):
    def test_baseline(self):
        """Test the baseline initialization."""
        box_env = TfEnv(DummyBoxEnv())
        DeterministicMLPBaseline(env_spec=box_env)
        GaussianMLPBaseline(env_spec=box_env)
