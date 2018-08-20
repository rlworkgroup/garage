import gym
import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.misc.overrides import overrides


class SlidingMemEnv(gym.Wrapper, Serializable):
    def __init__(
            self,
            env,
            n_steps=4,
            axis=0,
    ):
        super().__init__(env)

        self.n_steps = n_steps
        self.axis = axis
        self.buffer = None

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def reset_buffer(self, new_):
        assert self.axis == 0
        self.buffer = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.buffer[0:] = new_

    def add_to_buffer(self, new_):
        assert self.axis == 0
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[:1] = new_

    @property
    def observation_space(self):
        origin = self.env.observation_space
        low = origin.low[np.newaxis, ...]
        high = origin.high[np.newaxis, ...]
        return gym.spaces.Box(
            np.repeat(low, self.n_steps, axis=self.axis),
            np.repeat(high, self.n_steps, axis=self.axis),
            dtype=np.float32)

    @observation_space.setter
    def observation_space(self, obs_space):
        # Since gym.Wrapper sets this property in SlidingMemEnv.__init__(),
        # this void setter is required to avoid an AttributeError.
        return

    @overrides
    def reset(self):
        obs = self.env.reset()
        self.reset_buffer(obs)
        return self.buffer

    @overrides
    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        self.add_to_buffer(next_obs)
        return Step(self.buffer, reward, done, **info)
