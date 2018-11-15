"""Wrapper for resizing to (w, h) and converting frames to grayscale."""
import cv2
import gym
import numpy as np

from gym.spaces import Box


class ResizeAndGrayscaleWrapper(gym.core.Wrapper):
    def __init__(self, env, w, h, plot=False):
        super(ResizeAndGrayscaleWrapper, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, shape=[w, h], dtype=np.float32)
        self.w = w
        self.h = h
        self.plot = plot

    def _observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.w, self.h), interpolation=cv2.INTER_AREA)
        obs = obs.astype(np.float32) / 255.0
        if self.plot:
            self._display(obs)
        return obs

    def reset(self):
        return self._observation(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def _display(self, obs):
        cv2.imshow('image', np.squeeze(np.asarray(obs)))
        cv2.waitKey(0)
