import pickle

import numpy as np

from garage.envs.point_env import PointEnv
from tests.helpers import step_env


class TestPointEnv:

    def test_pickleable(self):
        env = PointEnv()
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        step_env(round_trip)
        env.close()
        round_trip.close()

    def test_does_not_modify_action(self):
        env = PointEnv()
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        assert a.all() == a_copy.all()
        env.close()

    def test_observation_space(self):
        env = PointEnv()
        obs_space = env.observation_space
        a = env.action_space.sample()
        obs, _, _, _ = env.step(a)
        assert obs_space.contains(obs)

    def test_reset(self):
        env = PointEnv()

        assert (env._point == np.array([0, 0])).all()

        a = env.action_space.sample()
        _ = env.step(a)
        env.reset()

        assert (env._point == np.array([0, 0])).all()

    def test_task(self):
        env = PointEnv()
        tasks = env.sample_tasks(5)
        assert len(tasks) == 5

        for task in tasks:
            env.set_task(task)
            assert (env._goal == task['goal']).all()

    def test_done(self):
        env = PointEnv()
        for _ in range(1000):
            _, _, done, _ = env.step(env._goal)
            if done:
                break
        else:
            assert False, 'Should report done'
