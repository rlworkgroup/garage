from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point
import numpy as np
import rospy

from rllab.spaces import Box


class TaskObject(object):
    def __init__(self, name, initial_pos, random_delta_range, resource=None):
        """
        Task Object interface
        :param name: str
        :param initial_pos: geometry_msgs.msg.Point
                object's original position
        :param random_delta_range: [float, float, float]
                positive, the range that would be used in sampling object' new
                start position for every episode. Set it as 0, if you want to keep the
                object's initial_pos for every episode.
        :param resource: str
                the model path(str) for simulation training or ros topic name for real robot training
        """
        self._name = name
        self._resource = resource
        self._initial_pos = Point(
            x=initial_pos[0], y=initial_pos[1], z=initial_pos[2])
        self._random_delta_range = random_delta_range

    @property
    def random_delta_range(self):
        return self._random_delta_range

    @property
    def name(self):
        return self._name

    @property
    def resource(self):
        return self._resource

    @property
    def initial_pos(self):
        return self._initial_pos

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        self._orientation = value


class TaskObjectManager(object):
    def __init__(self):
        """
        Users can use this interface to manage the objects in a specific task
        """
        self._manipulatables = []
        self._targets = []
        self._commons = []

    def add_target(self, task_object):
        self._targets.append(task_object)

    def add_manipulatable(self, task_object):
        self._manipulatables.append(task_object)

    def add_common(self, task_object):
        self._commons.append(task_object)

    @property
    def objects(self):
        return self._manipulatables + self._targets + self._commons

    @property
    def manipulatables(self):
        return self._manipulatables

    @property
    def targets(self):
        return self._targets

    def sub_gazebo(self):
        rospy.Subscriber('/gazebo/model_states', ModelStates,
                         self._gazebo_update_manipulatable_states)

    def _gazebo_update_manipulatable_states(self, data):
        model_states = data
        model_names = model_states.name

        for manipulatable in self._manipulatables:
            manipulatable_idx = model_names.index(manipulatable.name)
            manipulatable_pose = model_states.pose[manipulatable_idx]
            manipulatable.position = manipulatable_pose.position
            manipulatable.orientation = manipulatable_pose.orientation

    def get_manipulatables_observation(self):
        manipulatables_pos = np.array([])
        manipulatables_ori = np.array([])

        for manipulatable in self._manipulatables:
            pos = np.array([
                manipulatable.position.x, manipulatable.position.y,
                manipulatable.position.z
            ])
            ori = np.array([
                manipulatable.orientation.x, manipulatable.orientation.y,
                manipulatable.orientation.z, manipulatable.orientation.w
            ])
            manipulatables_pos = np.concatenate((manipulatables_pos, pos))
            manipulatables_ori = np.concatenate((manipulatables_ori, ori))

        achieved_goal = np.squeeze(manipulatables_pos)

        obs = np.concatenate((manipulatables_pos, manipulatables_ori))

        return {'obs': obs, 'achieved_goal': achieved_goal}

    @property
    def manipulatables_observation_space(self):
        return Box(
            -np.inf,
            np.inf,
            shape=self.get_manipulatables_observation()['obs'].shape)
