import gym

from garage.wrapper.resize_and_grayscale_wrapper import ResizeAndGrayscaleWrapper
from garage.wrapper.stack_frames_wrapper import StackFramesWrapper
from tests.fixtures import TfGraphTestCase


class TestWrappers(TfGraphTestCase):
    def test_resize_and_gray_scale_wrapper(self):
        env = ResizeAndGrayscaleWrapper(gym.make("Breakout-v0"), w=84, h=84)
        assert env.observation_space.shape == (84, 84)

    def test_stack_frames_wrapper(self):
        # StackFrameWrapper only supports input with channel=1
        env = StackFramesWrapper(
            ResizeAndGrayscaleWrapper(gym.make("Breakout-v0"), w=84, h=84),
            n_frames_stacked=4)

        env.reset()
        obs, _, _, _ = env.step(0)
        assert obs.shape == (84, 84, 4)
