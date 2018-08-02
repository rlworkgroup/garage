"""world without any objects."""

import collections

from geometry_msgs.msg import PoseStamped
import gym
import numpy as np

from garage.contrib.ros.worlds.world import World


class EmptyWorld(World):
    """Empty world class."""

    def __init__(self, moveit_scene, frame_id, simulated=False):
        """
        Users use this to manage world and get world state.

        :param moveit_scene: moveit scene
                Use this to add/Move/Remove objects in MoveIt!
        :param frame_id: string
                Use this to add/Move/Remove objects in MoveIt!
        :param simulated: Bool
                if simulated
        """
        self._simulated = simulated
        self._frame_id = frame_id
        self._moveit_scene = moveit_scene

    def initialize(self):
        """Use this to initialize the world."""
        # Add table to moveit
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self._frame_id
        pose_stamped.pose.position.x = 0.655
        pose_stamped.pose.position.y = 0
        # Leave redundant space
        pose_stamped.pose.position.z = -0.02
        pose_stamped.pose.orientation.x = 0
        pose_stamped.pose.orientation.y = 0
        pose_stamped.pose.orientation.z = 0
        pose_stamped.pose.orientation.w = 1.0
        self._moveit_scene.add_box('table', pose_stamped, (1.0, 0.9, 0.1))

    def reset(self):
        """Use this to reset the world."""
        pass

    def terminate(self):
        """Use this to terminate the world."""
        self._moveit_scene.remove_world_object('table')

    def get_observation(self):
        """Use this to get the observation from world."""
        achieved_goal = np.array([])

        obs = np.array([])

        Observation = collections.namedtuple('Observation',
                                             'obs achieved_goal')

        observation = Observation(obs=obs, achieved_goal=achieved_goal)

        return observation

    @property
    def observation_space(self):
        """Use this to get observation space."""
        return gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.get_observation().obs.shape,
            dtype=np.float32)
