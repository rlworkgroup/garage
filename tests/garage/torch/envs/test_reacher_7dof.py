import unittest

from garage.torch.envs.tdm.reacher_7dof_env import Reacher7DofFullGoal


class TestReacher7DOF(unittest.TestCase):
    def test_reacher_7dof(self):
        env = Reacher7DofFullGoal()
        env.reset()
        env.step(env.action_space.sample())
