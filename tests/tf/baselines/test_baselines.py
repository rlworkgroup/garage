"""
This script creates a test that fails when
garage.tf.baselines failed to initialize.
"""
import unittest

import tensorflow as tf
from tests.envs.dummy import DummyBoxEnv

import garage.misc.logger as logger
from garage.misc.tensorboard_output import TensorBoardOutput
from garage.tf.baselines import DeterministicMLPBaseline
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.envs import TfEnv


class TestTfBaselines(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session(graph=tf.Graph())
        self.sess.__enter__()
        logger._tensorboard = TensorBoardOutput()

    def tearDown(self):
        self.sess.close()

    def test_baseline(self):
        """Test the baseline initialization."""
        box_env = TfEnv(DummyBoxEnv())
        deterministic_mlp_baseline = DeterministicMLPBaseline(env_spec=box_env)
        gaussian_mlp_baseline = GaussianMLPBaseline(env_spec=box_env)
