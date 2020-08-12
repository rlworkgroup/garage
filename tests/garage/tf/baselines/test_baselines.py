"""
This script creates a test that fails when
garage.tf.baselines failed to initialize.
"""
import tensorflow as tf

from garage.envs import GymEnv
from garage.tf.baselines import ContinuousMLPBaseline, GaussianMLPBaseline

from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestTfBaselines(TfGraphTestCase):

    def test_baseline(self):
        """Test the baseline initialization."""
        box_env = GymEnv(DummyBoxEnv())
        deterministic_mlp_baseline = ContinuousMLPBaseline(env_spec=box_env)
        gaussian_mlp_baseline = GaussianMLPBaseline(env_spec=box_env)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        deterministic_mlp_baseline.get_param_values()
        gaussian_mlp_baseline.get_param_values()

        box_env.close()
