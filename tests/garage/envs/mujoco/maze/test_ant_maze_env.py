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
