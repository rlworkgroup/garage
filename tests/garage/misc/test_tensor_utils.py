"""
This script creates a test that tests functions in garage.misc.tensor_utils.
"""

import numpy as np

from garage.misc.tensor_utils import normalize_pixel_batch
from garage.tf.envs import TfEnv
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestTensorUtil:
    def test_normalize_pixel_patch(self):
        env = TfEnv(DummyDiscretePixelEnv())
        obs = env.reset()
        obs_normalized = normalize_pixel_batch(env, obs)
        expected = [ob / 255.0 for ob in obs]
        assert np.allclose(obs_normalized, expected)

    def test_normalize_pixel_patch_not_trigger(self):
        env = TfEnv(DummyBoxEnv())
        obs = env.reset()
        obs_normalized = normalize_pixel_batch(env, obs)
        assert np.array_equal(obs, obs_normalized)
