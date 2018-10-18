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
                a = env.action_space.sample()
                a_copy = a if isinstance(a, (int, float)) else a.copy()
                next_obs, _, done, _ = env.step(a)

                # Check for side effects
                if isinstance(a, (int, float)):
                    assert a == a_copy,\
                    "Action was modified by environment!"
                else:
                    assert a.all() == a_copy.all(),\
                    "Action was modified by environment!"

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
                a = env.action_space.sample()
                a_copy = a if isinstance(a, (int, float)) else a.copy()
                next_obs, reward, done, info = env.step(a)

                # Check for side effects
                if isinstance(a, (int, float)):
                    assert a == a_copy,\
                    "Action was modified by environment!"
                else:
                    assert a.all() == a_copy.all(),\
                    "Action was modified by environment!"

                assert (env.observation_space.flatten(next_obs).shape ==
                        env.observation_space.flat_dim)  # yapf: disable
                if done:
                    break
        env.close()
