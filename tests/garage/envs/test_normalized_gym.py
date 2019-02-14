import unittest

import gym

from garage.envs import normalize
from garage.tf.envs import TfEnv


class TestNormalizedGym(unittest.TestCase):
    def setUp(self):
        self.env = TfEnv(
            normalize(
                gym.make('Pendulum-v0'),
                normalize_reward=True,
                normalize_obs=True,
                flatten_obs=True))

    def tearDown(self):
        self.env.close()

    def test_does_not_modify_action(self):
        a = self.env.action_space.sample()
        a_copy = a
        self.env.reset()
        self.env.step(a)
        self.assertEquals(a, a_copy)

    def test_flatten(self):
        for _ in range(10):
            self.env.reset()
            for _ in range(5):
                self.env.render()
                action = self.env.action_space.sample()
                next_obs, _, done, _ = self.env.step(action)
                self.assertEqual(next_obs.shape,
                                 self.env.observation_space.low.shape)
                if done:
                    break

    def test_unflatten(self):
        for _ in range(10):
            self.env.reset()
            for _ in range(5):
                action = self.env.action_space.sample()
                next_obs, _, done, _ = self.env.step(action)
                self.assertEqual(
                    self.env.observation_space.flatten(next_obs).shape,
                    self.env.observation_space.flat_dim)
                if done:
                    break
