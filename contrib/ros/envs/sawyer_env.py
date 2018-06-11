"""
ROS environment for the Sawyer robot
Every specific sawyer task environment should inherit it.
"""
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
import numpy as np

from contrib.ros.envs.ros_env import RosEnv
from contrib.ros.robots.sawyer import Sawyer
from contrib.ros.util.common import rate_limited
from contrib.ros.util.gazebo import Gazebo
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.spaces import Box


class SawyerEnv(RosEnv, Serializable):
    """
    ROS Sawyer Env
    """

    def __init__(self,
                 task_obj_mgr,
                 robot,
                 has_object,
                 simulated=False,
                 obj_range=0.15):
        """
        :param task_obj_mgr: object
                User uses this to manage every other objects used in task except
                for robots.
        :param robot: object
                the robot interface for the environment
        :param has_object: bool
                if there is object in this experiment
        :param simulated: bool
                if the environment is for real robot or simulation
        :param obj_range: float
                the new initial object x,y position for the new training episode
                would be randomly sampled inside
                [initial_gripper_pos - obj_range,
                initial_gripper_pos + obj_range]
        """
        # run the superclass initialize function first
        Serializable.quick_init(self, locals())

        RosEnv.__init__(self)

        # Verify robot is enabled
        self._robot = robot

        if not self._robot.enabled:
            raise RuntimeError('The robot is not enabled!')
            # TODO (gh/74: Add initialize interface for robot)

        self._simulated = simulated

        self.task_obj_mgr = task_obj_mgr

        if self._simulated:
            self.gazebo = Gazebo()
            self.task_obj_mgr.sub_gazebo()
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass
        self.has_object = has_object
        self.obj_range = obj_range

        self._initial_setup()

    def shutdown(self):
        if self._simulated:
            # delete model
            for obj in self.task_obj_mgr.objects:
                self.gazebo.delete_gazebo_model(obj.name)
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass

    # Implementation for rllab env functions
    # -----------------------------------
    @rate_limited(100)
    def step(self, action):
        """
        Perform a step in gazebo. When end of episode
        is reached, reset() should be called to reset the environment's internal
        state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the
         previous action
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
        observation : the initial observation of the space. (Initial reward is
        assumed to be 0.)
        """
        self._robot.reset()

        self.goal = self.sample_goal()

        if self._simulated:
            target_idx = 0
            for target in self.task_obj_mgr.targets:
                self.gazebo.set_model_pose(
                    model_name=target.name,
                    new_pose=Pose(
                        position=Point(
                            x=self.goal[target_idx * 3],
                            y=self.goal[target_idx * 3 + 1],
                            z=self.goal[target_idx * 3 + 2])))
                target_idx += 1
            self._reset_sim()
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass
        obs = self.get_observation()
        initial_observation = obs['observation']

        return initial_observation

    def _reset_sim(self):
        """
        reset the simulation
        """
        # Randomize start position of object
        for manipulatable in self.task_obj_mgr.manipulatables:
            manipulatable_random_delta = np.zeros(2)
            while np.linalg.norm(manipulatable_random_delta) < 0.1:
                manipulatable_random_delta = np.random.uniform(
                    -manipulatable.random_delta_range,
                    manipulatable.random_delta_range,
                    size=2)
            self.gazebo.set_model_pose(
                manipulatable.name,
                new_pose=Pose(
                    position=Point(
                        x=manipulatable.initial_pos.x +
                        manipulatable_random_delta[0],
                        y=manipulatable.initial_pos.y +
                        manipulatable_random_delta[1],
                        z=manipulatable.initial_pos.z)))

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
    def _initial_setup(self):
        self._robot.reset()

        if self._simulated:
            # Generate the world
            # Load the model
            for obj in self.task_obj_mgr.objects:
                self.gazebo.load_gazebo_model(obj)
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
