"""
ROS environment for the Sawyer robot
Every specific sawyer task environment should inherit it.
"""

import os.path as osp

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Point
import numpy as np
import rospkg
import rospy

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc import logger
from rllab.spaces import Box

from contrib.ros.envs.ros_env import RosEnv
from contrib.ros.robots.sawyer import Sawyer
from contrib.ros.util.common import rate_limited
from contrib.ros.util.gazebo import Gazebo


def _goal_distance(goal_a, goal_b):
    """
    :param goal_a:
    :param goal_b:
    :return distance: distance between goal_a and goal_b
    """
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class SawyerEnv(RosEnv, Serializable):
    """
    ROS Sawyer Env
    """

    def __init__(self,
                 initial_robot_joint_pos,
                 robot_control_mode,
                 has_object,
                 distance_threshold,
                 initial_model_pos,
                 sparse_reward,
                 target_in_the_air,
                 simulated=False,
                 target_range=0.15,
                 obj_range=0.15):
        """
        :param initial_robot_joint_pos: {str: float}
                {'joint_name': value}
        :param robot control mode: str
                'effort'/'position'/'velocity'
        :param has_object: bool
                if there is object in this experiment
        :param distance_threshold: float
                When the distance from robot endpoint to the target point is smaller than this threshold
                the training episode is done.
                To calculate the sparse reward, if the distance is smaller than this threshold returns -1
                else returns 0.
        :param initial_model_pos: dict
                joint positions and object positions
        :param sparse_reward: bool
                if reward_type is 'sparse'
        :param target_in_the_air: bool
                set to see if the target can be in the air
        :param simulated: bool
                if the environment is for real robot or simulation
        :param target_range: float
                the new target position for the new training episode would be randomly sampled inside
                [initial_gripper_pos - target_range, initial_gripper_pos + target_range]
        :param obj_range: float
                the new initial object x,y position for the new training episode would be randomly sampled inside
                [initial_gripper_pos - obj_range, initial_gripper_pos + obj_range]
        """
        # run the superclass initialize function first
        Serializable.quick_init(self, locals())

        RosEnv.__init__(self)

        # Verify robot is enabled
        self._robot = Sawyer(
            initial_robot_joint_pos, control_mode=robot_control_mode)
        if not self._robot.enabled:
            raise RuntimeError('The robot is not enabled!')
            # TODO (gh/74: Add initialize interface for robot)

        self._simulated = simulated
        if self._simulated:
            rospy.Subscriber('/gazebo/model_states', ModelStates,
                             self.update_model_states)
            self._gazebosrv = Gazebo()
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass
        self._initial_model_pos = initial_model_pos
        self.has_object = has_object
        self.distance_threshold = distance_threshold
        self.sparse_reward = sparse_reward
        self.target_range = target_range
        self.target_in_the_air = target_in_the_air
        self.obj_range = obj_range

        rospy.on_shutdown(self._shutdown)

        self.initial_setup(self._initial_model_pos)

    def _shutdown(self):
        logger.log('Exiting...')
        if self._simulated:
            # delete model
            logger.log('Delete gazebo models...')
            self._gazebosrv.delete_gazebo_model(model_name='cafe_table')
            self._gazebosrv.delete_gazebo_model(model_name='block')
            self._gazebosrv.delete_gazebo_model(model_name='target')

    def update_model_states(self, data):
        if self._simulated:
            self.model_states = data
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass

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
        d = _goal_distance(achieved_goal, goal)
        if d < self.distance_threshold:
            return 88
        else:
            if self.sparse_reward:
                return -1.
            else:
                return -d

    # Implementation for rllab env functions
    # -----------------------------------
    @rate_limited(100)
    def step(self, action):
        """
        Perform a step in gazebo. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        self._robot.send_command(action)

        obs = self._get_obs()

        reward = self.reward(obs['achieved_goal'], self.goal)
        done = _goal_distance(obs['achieved_goal'],
                              self.goal) < self.distance_threshold
        next_observation = obs['observation']
        return Step(observation=next_observation, reward=reward, done=done)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.goal = self.sample_goal()

        if self._simulated:
            self._gazebosrv.set_model_pose(
                model_name='target',
                new_pose=Pose(
                    position=Point(
                        x=self.goal[0], y=self.goal[1], z=self.goal[2])),
                reference_frame='world')
            self._reset_sim()
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass
        obs = self._get_obs()
        initial_observation = obs['observation']

        return initial_observation

    # yapf: disable
    def _reset_sim(self):
        """
        reset the simulation
        """
        self._robot.reset()

        # Randomize start position of object
        if self.has_object:
            object_xypos = self.initial_gripper_pos[:2]
            while np.linalg.norm(object_xypos -
                                 self.initial_gripper_pos[:2]) < 0.1:
                object_random_delta = np.random.uniform(-self.obj_range,
                                                        self.obj_range,
                                                        size=2)
                object_xypos = self.initial_gripper_pos[:2] + object_random_delta
            self._gazebosrv.set_model_pose(
                model_name='block',
                new_pose=Pose(
                    position=Point(
                        x=object_xypos[0],
                        y=object_xypos[1],
                        z=self._initial_model_pos['object0'][2])),
                reference_frame='world')
    # yapf: enable

    @property
    def action_space(self):
        return self._robot.action_space

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        return Box(-np.inf, np.inf, shape=self._get_obs()['observation'].shape)

    # ================================================
    # Functions that gazebo env asks to implement
    # ================================================
    def initial_setup(self, initial_model_pos):
        # Extract information for sampling goals
        gripper_pose = self._robot.gripper_pose
        self.initial_gripper_pos = np.array(gripper_pose['position'])
        self.initial_gripper_ori = np.array(gripper_pose['orientation'])
        if self.has_object:
            self.height_offset = initial_model_pos['object0'][2]

        self._robot.reset()

        if self._simulated:
            # Generate the world
            # Load the model
            table_model_pose = Pose(
                position=Point(initial_model_pos['table0'][
                    0], initial_model_pos['table0'][1], initial_model_pos[
                        'table0'][2]))
            table_model_path = osp.join(
                rospkg.RosPack().get_path('sawyer_sim_examples'), 'models',
                'cafe_table/model.sdf')
            self._gazebosrv.load_gazebo_sdf_model(
                model_name='cafe_table',
                model_pose=table_model_pose,
                model_path=table_model_path)
            block_model_pose = Pose(
                position=Point(initial_model_pos['object0'][
                    0], initial_model_pos['object0'][1], initial_model_pos[
                        'object0'][2]))
            block_model_path = osp.join(
                rospkg.RosPack().get_path('sawyer_sim_examples'), 'models',
                'block/model.urdf')
            self._gazebosrv.load_gazebo_urdf_model(
                model_name='block',
                model_pose=block_model_pose,
                model_path=block_model_path)
            target_model_pose = Pose(
                position=Point(gripper_pose['position'][0], gripper_pose[
                    'position'][1], gripper_pose['position'][2]))
            target_model_path = osp.join(
                rospkg.RosPack().get_path('sawyer_sim_examples'), 'models',
                'target/model.sdf')
            self._gazebosrv.load_gazebo_sdf_model(
                model_name='target',
                model_pose=target_model_pose,
                model_path=target_model_path)
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass

    def sample_goal(self):
        """
        Sample goals
        :return: the new sampled goal
        """
        if self.has_object:
            random_goal_delta = np.random.uniform(
                -self.target_range, self.target_range, size=3)
            goal = self.initial_gripper_pos[:3] + random_goal_delta
            goal[2] = self.height_offset
            # If the target can't set to be in the air like pick and place task
            # it has 50% probability to be in the air
            if self.target_in_the_air and np.random.uniform() < 0.5:
                goal[2] += np.random.uniform(0, 0.45)
        else:
            random_goal_delta = np.random.uniform(-0.15, 0.15, size=3)
            goal = self.initial_gripper_pos[:3] + random_goal_delta
        return goal

    # -------------------------------------------------------------------------------------------
    def _get_obs(self):
        robot_obs = self._robot.get_observation()

        # gazebo data message
        if self.has_object:
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
        else:
            object_pos = np.array([])
            object_ori = np.array([])

        if not self.has_object:
            achieved_goal = self._robot.gripper_pose
        else:
            achieved_goal = np.squeeze(object_pos)

        obs = np.concatenate((robot_obs, object_pos, object_ori))

        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': self.goal
        }
