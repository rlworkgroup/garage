import unittest

from garage.envs.mujoco.gather.gather_env import GatherEnv


class TestGatherEnv(unittest.TestCase):
    def test_must_subclass(self):
        # GatherEnv needs to be subclassed and subclasses specify a model class
        with self.assertRaises(NotImplementedError):
            env = GatherEnv()
