"""Reacher task for the sawyer robot."""

import collections

import gym
import moveit_commander
import numpy as np

from garage.contrib.ros.envs.sawyer.sawyer_env import SawyerEnv
from garage.contrib.ros.robots import Sawyer
from garage.contrib.ros.worlds import EmptyWorld
from garage.core import Serializable


class ReacherEnv(SawyerEnv, Serializable):
    """Reacher Environment."""

    def __init__(self,
                 initial_goal,
                 initial_joint_pos,
                 sparse_reward=False,
                 simulated=False,
                 distance_threshold=0.05,
                 target_range=0.15,
                 robot_control_mode='position'):
        """
        Reacher Environment.

        :param initial_goal: np.array
                        the initial goal for the task
        :param initial_joint_pos: dict
                        the initial joint angles for the sawyer
        :param sparse_reward: Bool
                        if use sparse reward
        :param simulated: Bool
                        if run simulated experiment
        :param distance_threshold: float
                        threshold for whether experiment is done
        :param target_range: float
                        delta range the goal is randomized
        :param robot_control_mode: string
                        robot control mode: 'position' or 'velocity'
                        or 'effort'
        """
        Serializable.quick_init(self, locals())

        self._distance_threshold = distance_threshold
        self._target_range = target_range
        self._sparse_reward = sparse_reward
        self.initial_goal = initial_goal
        self.goal = self.initial_goal.copy()
        self.simulated = simulated

        # Initialize moveit to get safety check
        self._moveit_robot = moveit_commander.RobotCommander()
        self._moveit_scene = moveit_commander.PlanningSceneInterface()
        self._moveit_group_name = 'right_arm'
        self._moveit_group = moveit_commander.MoveGroupCommander(
            self._moveit_group_name)

        self._robot = Sawyer(
            initial_joint_pos=initial_joint_pos,
            control_mode=robot_control_mode,
            moveit_group=self._moveit_group_name)
        self._world = EmptyWorld(self._moveit_scene,
                                 self._moveit_robot.get_planning_frame(),
                                 simulated)

        SawyerEnv.__init__(self, simulated=simulated)

    @property
    def observation_space(self):
        """
        Return a Space object.

        :return: the observation space
        """
        return gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.get_observation().observation.shape,
            dtype=np.float32)

    def sample_goal(self):
        """
        Sample goals.

        :return: the new sampled goal
        """
        goal = self.initial_goal.copy()

        random_goal_delta = np.random.uniform(
            -self._target_range, self._target_range, size=2)
        goal[:2] += random_goal_delta

        return goal

    def get_observation(self):
        """
        Get Observation.

        :return observation: dict
                    {'observation': obs,
                     'achieved_goal': achieved_goal,
                     'desired_goal': self.goal}
        """
        obs = self._robot.get_observation()

        robot_gripper_pos = self._robot.gripper_pose['position']

        achieved_goal = np.array(
            [robot_gripper_pos.x, robot_gripper_pos.y, robot_gripper_pos.z])

        Observation = collections.namedtuple(
            'Observation', 'observation achieved_goal desired_goal')

        observation = Observation(
            observation=obs,
            achieved_goal=achieved_goal,
            desired_goal=self.goal)

        return observation

    def reward(self, achieved_goal, goal):
        """
        Compute the reward for current step.

        :param achieved_goal: np.array
                    the current gripper's position or object's
                    position in the current training episode.
        :param goal: np.array
                    the goal of the current training episode, which mostly
                    is the target position of the object or the position.
        :return reward: float
                    if sparse_reward, the reward is -1, else the
                    reward is minus distance from achieved_goal to
                    our goal. And we have completion bonus for two
                    kinds of types.
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
        Compute distance between achieved goal and goal.

        :param goal_a:
        :param goal_b:
        :return distance: distance between goal_a and goal_b
        """
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def done(self, achieved_goal, goal):
        """
        If done.

        :return if_done: bool
                    if current episode is done:
        """
        if not self._robot.safety_check():
            done = True
        else:
            done = self._goal_distance(achieved_goal,
                                       goal) < self._distance_threshold
        return done
