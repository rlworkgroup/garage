import pickle

import numpy as np

from garage.envs import PointEnv
from garage.envs.normalized_env import NormalizedEnv

from tests.helpers import step_env


class TestNormalizedEnv:

    def test_pickleable(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = NormalizedEnv(inner_env, scale_reward=10.)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip._scale_reward == env._scale_reward
        assert np.array_equal(round_trip._env._goal, env._env._goal)
        step_env(round_trip, visualize=False)
        env.close()
        round_trip.close()

    def test_does_not_modify_action(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = NormalizedEnv(inner_env, scale_reward=10.)
        a = env.action_space.high + 1.
        a_copy = a
        env.reset()
        env.step(a)
        assert np.array_equal(a, a_copy)
        env.close()

    def test_visualization(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = NormalizedEnv(inner_env)

        env.visualize()
        env.reset()
        assert inner_env.render_modes == env.render_modes
        mode = inner_env.render_modes[0]
        assert inner_env.render(mode) == env.render(mode)

    def test_no_flatten_obs(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = NormalizedEnv(inner_env, flatten_obs=False)
        obs = env.reset()[0]

        assert obs.shape == env.observation_space.shape
