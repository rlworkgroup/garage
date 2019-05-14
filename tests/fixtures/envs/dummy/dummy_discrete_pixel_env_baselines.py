import gym
import numpy as np

from tests.fixtures.envs.dummy import DummyEnv


class LazyFrames(object):
    def __init__(self, frames):
        """
        LazyFrames class from baselines.

        Openai baselines use this class for FrameStack environment
        wrapper. It is used for testing garage.envs.wrappers.AtariEnv.
        garge.envs.wrapper.AtariEnv is used when algorithms are trained
        using baselines wrappers, e.g. during benchmarking.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out


class DummyDiscretePixelEnvBaselines(DummyEnv):
    """
    A dummy discrete pixel environment.

    This environment is for testing garge.envs.wrapper.AtariEnv.
    """

    def __init__(self):
        super().__init__(random=False, obs_dim=(10, 10, 3), action_dim=5)
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=self._obs_dim, dtype=np.uint8)

    @property
    def observation_space(self):
        """Return an observation space."""
        return self._observation_space

    @property
    def action_space(self):
        """Return an action space."""
        return gym.spaces.Discrete(self._action_dim)

    def step(self, action):
        """gym.Env step function."""
        obs = self.observation_space.sample()
        return LazyFrames([obs]), 0, True, dict()

    def reset(self, **kwargs):
        """gym.Env reset function."""
        obs = np.ones(self._obs_dim, dtype=np.uint8)
        return LazyFrames([obs])
