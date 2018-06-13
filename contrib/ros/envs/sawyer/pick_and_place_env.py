"""
Pick-and-place task for the sawyer robot
"""
import collections

import numpy as np

from contrib.ros.envs.sawyer.sawyer_env import SawyerEnv
from contrib.ros.robots.sawyer import Sawyer
from contrib.ros.worlds.block_world import BlockWorld
from garage.core import Serializable
from garage.spaces import Box


class PickAndPlaceEnv(SawyerEnv, Serializable):
    def __init__(self,
                 initial_goal,
                 initial_joint_pos,
                 sparse_reward=True,
                 simulated=False,
                 distance_threshold=0.05,
                 target_range=0.15,
                 robot_control_mode='position',
                 step_freq=100):
        Serializable.quick_init(self, locals())

        self._distance_threshold = distance_threshold
        self._target_range = target_range
        self._sparse_reward = sparse_reward
        self.initial_goal = initial_goal
        self.goal = self.initial_goal.copy()

        self._sawyer = Sawyer(
            initial_joint_pos=initial_joint_pos,
            control_mode=robot_control_mode)
        self._block_world = BlockWorld(simulated)

        SawyerEnv.__init__(
            self,
            simulated=simulated,
            robot=self._sawyer,
            world=self._block_world,
            step_freq=step_freq)

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        return Box(
            -np.inf, np.inf, shape=self.get_observation().observation.shape)

    def sample_goal(self):
        """
        Sample goals
        :return: the new sampled goal
        """
        goal = self.initial_goal.copy()

        random_goal_delta = np.random.uniform(
            -self._target_range, self._target_range, size=2)
        goal[:2] += random_goal_delta

        return goal

    def get_observation(self):
        """
        Get Observation
        :return observation: dict
                    {'observation': obs,
                     'achieved_goal': achieved_goal,
                     'desired_goal': self.goal}
        """
        robot_obs = self._sawyer.get_observation()

        world_obs = self._block_world.get_observation()

        obs = np.concatenate((robot_obs, world_obs.obs))

        Observation = collections.namedtuple(
            'Observation', 'observation achieved_goal desired_goal')

        observation = Observation(
            observation=obs,
            achieved_goal=world_obs.achieved_goal,
            desired_goal=self.goal)

        return observation

    def reward(self, achieved_goal, goal):
        """
        Compute the reward for current step.
        :param achieved_goal: the current gripper's position or object's position in the current training episode.
        :param goal: the goal of the current training episode, which mostly is the target position of the object or the
                     position.
        :return reward: float
                    if sparse_reward, the reward is -1, else the reward is minus distance from achieved_goal to
                    our goal. And we have completion bonus for two kinds of types.
        """
        d = self._goal_distance(achieved_goal, goal)
        if d < self._distance_threshold:
            return 100
        else:
            if self._sparse_reward:
                return -1.
            else:
                return -d

    def _goal_distance(self, goal_a, goal_b):
        """
        :param goal_a:
        :param goal_b:
        :return distance: distance between goal_a and goal_b
        """
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def done(self, achieved_goal, goal):
        """
        :return if_done: bool
                    if current episode is done:
        """
        return self._goal_distance(achieved_goal,
                                   goal) < self._distance_threshold
