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


class SawyerEnv(RosEnv, Serializable):
    """
    ROS Sawyer Env
    """

    def __init__(self,
                 initial_robot_joint_pos,
                 robot_control_mode,
                 has_object,
                 initial_model_pos,
                 simulated=False,
                 obj_range=0.15):
        """
        :param initial_robot_joint_pos: {str: float}
                {'joint_name': value}
        :param robot control mode: str
                'effort'/'position'/'velocity'
        :param has_object: bool
                if there is object in this experiment
        :param initial_model_pos: dict
                joint positions and object positions
        :param simulated: bool
                if the environment is for real robot or simulation
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

        obs = self.get_observation()

        reward = self.reward(obs['achieved_goal'], self.goal)
        done = self.done(obs['achieved_goal'], self.goal)
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
        obs = self.get_observation()
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
        return Box(
            -np.inf, np.inf, shape=self.get_observation()['observation'].shape)

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

    def done(self, achieved_goal, goal):
        """
        :return if done: bool
        """
        raise NotImplementedError

    def reward(self, achieved_goal, goal):
        """
        Compute the reward for current step.
        """
        raise NotImplementedError

    def sample_goal(self):
        """
        Samples a new goal and returns it.
        """
        raise NotImplementedError

    def get_observation(self):
        """
        Get observation
        """
        raise NotImplementedError
