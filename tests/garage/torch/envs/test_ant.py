import unittest

from garage.torch.envs.tdm.ant_env import GoalXYPosAnt
from garage.torch.envs.tdm.ant_env import GoalXYPosAndVelAnt


class TestAnt(unittest.TestCase):
    def test_pos_vel_ant(self):
        env = GoalXYPosAndVelAnt(
            goal_dim_weights=[0.1, 0.1, 0.9, 0.9],
            speed_weight=None,
        )
        env.reset()
        env.step(env.action_space.sample())

    def test_pos_ant(self):
        env = GoalXYPosAnt(max_distance=6)
        env.reset()
        env.step(env.action_space.sample())
