"""
Pick-and-place task for the sawyer robot
"""

import numpy as np

from rllab.core.serializable import Serializable

from contrib.ros.envs import sawyer_env

INITIAL_MODEL_POS = {
    'table0': [0.75, 0.0, 0.0],
    'object0': [0.5725, 0.1265, 0.80]
}

INITIAL_ROBOT_JOINT_POS = {
    'right_j0': -0.041662954890248294,
    'right_j1': -1.0258291091425074,
    'right_j2': 0.0293680414401436,
    'right_j3': 2.17518162913313,
    'right_j4': -0.06703022873354225,
    'right_j5': 0.3968371433926965,
    'right_j6': 1.7659649178699421,
}


class PickAndPlaceEnv(sawyer_env.SawyerEnv, Serializable):
    def __init__(self,
                 sparse_reward=True,
                 distance_threshold=0.05,
                 target_range=0.15,
                 target_in_the_air=False):
        Serializable.quick_init(self, locals())

        self._distance_threshold = distance_threshold
        self._target_range = target_range
        self._sparse_reward = sparse_reward
        self._target_in_the_air = target_in_the_air

        sawyer_env.SawyerEnv.__init__(
            self,
            initial_robot_joint_pos=INITIAL_ROBOT_JOINT_POS,
            robot_control_mode='position',
            has_object=True,
            initial_model_pos=INITIAL_MODEL_POS,
            simulated=True,
            obj_range=0.15)

    def sample_goal(self):
        """
        Sample goals
        :return: the new sampled goal
        """
        random_goal_delta = np.random.uniform(
            -self._target_range, self._target_range, size=3)
        goal = self.initial_gripper_pos[:3] + random_goal_delta
        goal[2] = self.height_offset
        # If the target can't set to be in the air like pick and place task
        # it has 50% probability to be in the air
        if self._target_in_the_air and np.random.uniform() < 0.5:
            goal[2] += np.random.uniform(0, 0.45)

        return goal

    def get_observation(self):
        robot_obs = self._robot.get_observation()

        # gazebo data message
        object_pos = np.array([])
        object_ori = np.array([])
        if self.model_states is not None:
            model_names = self.model_states.name
            object_idx = model_names.index('block')
            object_pose = self.model_states.pose[object_idx]
            object_pos = np.array([
                object_pose.position.x, object_pose.position.y,
                object_pose.position.z
            ])
            object_ori = np.array([
                object_pose.orientation.x, object_pose.orientation.y,
                object_pose.orientation.z, object_pose.orientation.w
            ])

        achieved_goal = np.squeeze(object_pos)

        obs = np.concatenate((robot_obs, object_pos, object_ori))

        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': self.goal
        }

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
