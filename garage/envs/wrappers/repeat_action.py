"""Repeat action wrapper for gym.Env."""
import gym


class RepeatAction(gym.Wrapper):
    """Given an action, the gym.Env executes that action for n frames."""

    def __init__(self, env, n_frame_to_repeat):
        """
        Repeat action wrapper.

        Args:
            env: gym.Env to wrap.
            n_frame_to_repeat: number of frames to repeat action.
        """
        super().__init__(env)
        self.n_frame_to_repeat = n_frame_to_repeat

    def step(self, action):
        """gym.Env step."""
        for i in range(self.n_frame_to_repeat):
            obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        """gym.Env reset."""
        return self.env.reset()
