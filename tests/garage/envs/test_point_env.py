import pickle

import numpy as np
import pytest

from garage.envs.point_env import PointEnv


class TestPointEnv:

    def test_pickleable(self):
        env = PointEnv()
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        round_trip.reset()
        round_trip.step(round_trip.action_space.sample())
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
        env.reset()
        obs_space = env.observation_space
        a = env.action_space.sample()
        obs = env.step(a).observation
        assert obs_space.contains(obs)

    def test_reset(self):
        env = PointEnv()
        env.reset()

        assert (env._point == np.array([0, 0])).all()

        a = env.action_space.sample()
        env.step(a)
        env.reset()

        assert (env._point == np.array([0, 0])).all()

    def test_task(self):
        env = PointEnv()
        env.reset()
        tasks = env.sample_tasks(5)
        assert len(tasks) == 5

        for task in tasks:
            env.set_task(task)
            assert (env._goal == task['goal']).all()

    def test_done(self):
        env = PointEnv()
        env.reset()
        for _ in range(1000):
            done = env.step(env._goal).terminal
            if done:
                break
        else:
            assert False, 'Should report done'

    def test_visualization(self):
        env = PointEnv()
        assert env.render_modes == ['ascii']
        env.reset()
        assert env.render('ascii') == f'Point: {env._point}, Goal: {env._goal}'

        env.visualize()
        env.step(env.action_space.sample())

    def test_catch_no_reset(self):
        env = PointEnv()
        with pytest.raises(RuntimeError, match='reset()'):
            env.step(env.action_space.sample())
