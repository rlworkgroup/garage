import unittest

from garage.envs.mujoco.mujoco_env import MujocoEnv


class TestMujocoEnv(unittest.TestCase):
    def test_must_subclass(self):
        # MujocoEnv needs to be subclassed and subclasses specify an XML file
        # in the class member FILE
        with self.assertRaises(NotImplementedError):
            env = MujocoEnv()
