import unittest

import gym

from garage.envs import normalize
from garage.theano.envs import TheanoEnv


class TestNormalizedGym(unittest.TestCase):
    def test_flatten(self):
        env = TheanoEnv(
            normalize(
                gym.make('Pendulum-v0'),
                normalize_reward=True,
                normalize_obs=True,
                flatten_obs=True))
        for i in range(10):
            env.reset()
            for e in range(5):
                env.render()
                action = env.action_space.sample()
                next_obs, reward, done, info = env.step(action)
                assert next_obs.shape == env.observation_space.low.shape
                if done:
                    break
        env.close()

    def test_unflatten(self):
        env = TheanoEnv(
            normalize(
                gym.make('Blackjack-v0'),
                normalize_reward=True,
                normalize_obs=True,
                flatten_obs=False))
        for i in range(10):
            env.reset()
            for e in range(5):
                action = env.action_space.sample()
                next_obs, reward, done, info = env.step(action)
                assert (env.observation_space.flatten(next_obs).shape ==
                        env.observation_space.flat_dim)  # yapf: disable
                if done:
                    break
        env.close()
