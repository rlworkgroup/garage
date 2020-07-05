import gym.spaces
import numpy as np
import pytest

from garage.envs.wrappers import Grayscale

from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestGrayscale:

    def setup_method(self):
        self.env = DummyDiscretePixelEnv(random=False)
        self.env_g = Grayscale(DummyDiscretePixelEnv(random=False))

    def teardown_method(self):
        self.env.close()
        self.env_g.close()

    def test_grayscale_invalid_environment_type(self):
        with pytest.raises(ValueError):
            self.env.observation_space = gym.spaces.Discrete(64)
            Grayscale(self.env)

    def test_grayscale_invalid_environment_shape(self):
        with pytest.raises(ValueError):
            self.env.observation_space = gym.spaces.Box(low=0,
                                                        high=255,
                                                        shape=(4, ),
                                                        dtype=np.uint8)
            Grayscale(self.env)

    def test_grayscale_observation_space(self):
        assert self.env_g.observation_space.shape == (
            self.env.observation_space.shape[:-1])

    def test_grayscale_reset(self):
        """
        RGB to grayscale conversion using scikit-image.

        Weights used for conversion:
        Y = 0.2125 R + 0.7154 G + 0.0721 B

        Reference:
        http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2grey
        """
        grayscale_output = np.round(
            np.dot(self.env.reset()[:, :, :3],
                   [0.2125, 0.7154, 0.0721])).astype(np.uint8)
        np.testing.assert_array_almost_equal(grayscale_output,
                                             self.env_g.reset())

    def test_grayscale_step(self):
        self.env.reset()
        self.env_g.reset()
        obs, _, _, _ = self.env.step(1)
        obs_g, _, _, _ = self.env_g.step(1)

        grayscale_output = np.round(
            np.dot(obs[:, :, :3], [0.2125, 0.7154, 0.0721])).astype(np.uint8)
        np.testing.assert_array_almost_equal(grayscale_output, obs_g)
