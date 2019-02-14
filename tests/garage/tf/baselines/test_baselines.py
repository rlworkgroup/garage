"""
This script creates a test that fails when
garage.tf.baselines failed to initialize.
"""
import tensorflow as tf

from garage.envs.wrappers import Resize
from garage.tf.baselines import DeterministicMLPBaseline
from garage.tf.baselines import GaussianConvBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.envs.dummy import DummyDiscrete2DEnv


class TestTfBaselines(TfGraphTestCase):
    def test_baseline(self):
        """Test the baseline initialization."""
        box_env = TfEnv(DummyBoxEnv())
        deterministic_mlp_baseline = DeterministicMLPBaseline(env_spec=box_env)
        gaussian_mlp_baseline = GaussianMLPBaseline(env_spec=box_env)

        discrete_env = TfEnv(Resize(DummyDiscrete2DEnv(), width=64, height=64))
        gaussian_conv_baseline = GaussianConvBaseline(
            env_spec=discrete_env,
            regressor_args=dict(
                conv_filters=[32, 32],
                conv_filter_sizes=[1, 1],
                conv_strides=[1, 1],
                conv_pads=["VALID", "VALID"],
                hidden_sizes=(32, 32)))

        self.sess.run(tf.global_variables_initializer())
        deterministic_mlp_baseline.get_param_values(trainable=True)
        gaussian_mlp_baseline.get_param_values(trainable=True)
        gaussian_conv_baseline.get_param_values(trainable=True)

        box_env.close()
