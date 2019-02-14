import pickle
import unittest

from garage.envs.mujoco.maze.ant_maze_env import AntMazeEnv
from tests.helpers import step_env


class TestAntMazeEnv(unittest.TestCase):
    def test_pickleable(self):
        env = AntMazeEnv(n_bins=2)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip._n_bins == env._n_bins
        step_env(round_trip)
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        env = AntMazeEnv(n_bins=2)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
        env.close()
