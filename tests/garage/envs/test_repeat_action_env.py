import unittest
from unittest import mock

from gym.spaces import Box
import numpy as np

from garage.envs.wrappers import RepeatAction
from garage.misc.overrides import overrides


class TestRepeatAction(unittest.TestCase):
    @overrides
    def setUp(self):
        self.shape = (16, )
        self.env = mock.Mock()
        self.env.observation_space = Box(
            low=0, high=255, shape=self.shape, dtype=np.float32)
        self.env.reset.return_value = np.zeros(self.shape)
        self.env.step.side_effect = self._step

        self.env_r = RepeatAction(self.env, n_frame_to_repeat=4)

        self.obs = self.env.reset()
        self.obs_r = self.env_r.reset()

    def _step(self, action):
        def generate():
            for i in range(0, 255):
                yield np.full(self.shape, i)

        generator = generate()

        return next(generator), 0, False, dict()

    def test_repeat_action_reset(self):
        np.testing.assert_array_equal(self.obs, self.obs_r)

    def test_repeat_action_step(self):
        obs_repeat, _, _, _ = self.env_r.step(1)
        for i in range(4):
            obs, _, _, _ = self.env.step(1)

        np.testing.assert_array_equal(obs, obs_repeat)
