"""Gym env wrapper for repeating action for n frames."""
import gym


class RepeatActionWrapper(gym.core.Wrapper):
    def __init__(self, env, frame_to_repeat):
        super(RepeatActionWrapper, self).__init__(env)
        self.frame_to_repeat = frame_to_repeat

    def step(self, action):
        for i in range(self.frame_to_repeat):
            obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()
