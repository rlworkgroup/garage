import numpy as np
import pickle
import unittest

from garage.envs.point_env import PointEnv
from garage.envs.multitask_env import MultitaskEnv
from tests.helpers import step_env


class TestMultitaskEnv(unittest.TestCase):
    def test_pickleable(self):
        inner_env = PointEnv()
        tasks = [inner_env.task_space.sample() for _ in range(2)]
        env = MultitaskEnv(wrapped_env=inner_env, tasks=tasks)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert np.all(round_trip.tasks) == np.all(env.tasks)
        assert round_trip.env.goal.all() == env.env.goal.all()
        step_env(round_trip)

    def test_does_not_modify_action(self):
        inner_env = PointEnv()
        tasks = [inner_env.task_space.sample() for _ in range(2)]
        env = MultitaskEnv(wrapped_env=inner_env, tasks=tasks)
        a = env.action_space.sample()
        a_copy = a
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())

    def test_reset_task(self):
        inner_env = PointEnv()
        tasks = [inner_env.task_space.sample() for _ in range(2)]
        env = MultitaskEnv(wrapped_env=inner_env, tasks=tasks)
        # Round robin
        env.reset_task()
        assert env.running_task_id == 1
        # Random
        env.task_sample_method = "random"
        env.reset_task()
        # Setting task by hand
        env.reset_task(0)
        assert env.running_task_id == 0
