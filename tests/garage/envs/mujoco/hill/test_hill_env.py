import unittest

from garage.envs.mujoco.hill.hill_env import HillEnv


class TestHillEnv(unittest.TestCase):
    def test_must_subclass(self):
        # HillEnv needs to be subclassed and subclasses specify a model class
        with self.assertRaises(NotImplementedError):
            env = HillEnv()
