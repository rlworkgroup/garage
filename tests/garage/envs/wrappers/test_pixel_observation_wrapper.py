import gym
import numpy as np
import pytest

from garage.envs.wrappers import PixelObservationWrapper


@pytest.mark.mujoco
class TestPixelObservationWrapper:

    def setup_method(self):
        self.env = gym.make('InvertedDoublePendulum-v2')
        self.pixel_env = PixelObservationWrapper(self.env)

    def teardown_method(self):
        self.env.close()
        self.pixel_env.close()

    def test_pixel_env_invalid_environment_type(self):
        with pytest.raises(ValueError):
            self.env.observation_space = gym.spaces.Discrete(64)
            PixelObservationWrapper(self.env)

    def test_pixel_env_observation_space(self):
        assert isinstance(self.pixel_env.observation_space, gym.spaces.Box)
        assert (self.pixel_env.observation_space.low == 0).all()
        assert (self.pixel_env.observation_space.high == 255).all()

    def test_pixel_env_reset(self):
        obs = self.pixel_env.reset()
        assert (obs <= 255.).all() and (obs >= 0.).all()
        assert isinstance(obs, np.ndarray)

    def test_pixel_env_step(self):
        self.pixel_env.reset()
        action = np.full(self.pixel_env.action_space.shape, 0)
        obs, _, _, _ = self.pixel_env.step(action)
        assert (obs <= 255.).all() and (obs >= 0.).all()
