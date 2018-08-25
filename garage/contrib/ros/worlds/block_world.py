import collections
import os.path as osp

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, \
                              TransformStamped
import gym
import numpy as np
import rospy

from garage.contrib.ros.worlds.gazebo import Gazebo
from garage.contrib.ros.worlds.world import World
import garage.misc.logger as logger
try:
    from garage.config import VICON_TOPICS
except ImportError:
    raise NotImplementedError(
        "Please set VICON_TOPICS in garage/config_personal.py! "
        "example 1:"
        "   VICON_TOPICS = ['<vicon_topic_name>']"
        "example 2:"
        "   # if you are not using real robot and vicon system"
        "   VICON_TOPICS = []")


class Block:
    def __init__(self, name, initial_pos, random_delta_range, resource=None):
        """
        Task Object interface
        :param name: str
        :param initial_pos: geometry_msgs.msg.Point
                object's original position
        :param random_delta_range: [float, float, float]
                positive, the range that would be used in
                sampling object' new start position for every episode.
                Set it as 0, if you want to keep the
                object's initial_pos for every episode.
        :param resource: str
                the model path(str) for simulation training or ros
                topic name for real robot training
        """
        self._name = name
        self._resource = resource
        self._initial_pos = Point(
            x=initial_pos[0], y=initial_pos[1], z=initial_pos[2])
        self._random_delta_range = random_delta_range
        self._position = Point(
            x=initial_pos[0], y=initial_pos[1], z=initial_pos[2])
        self._orientation = Quaternion(x=0., y=0., z=0., w=1.)

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


class BlockWorld(World):
    def __init__(self, moveit_scene, frame_id, simulated=False):
        """
        Users use this to manage world and get world state.
        """
        self._blocks = []
        self._simulated = simulated
        self._block_states_subs = []
        self._moveit_scene = moveit_scene
        self._frame_id = frame_id

    def initialize(self):
        if self._simulated:
            Gazebo.load_gazebo_model(
                'table',
                Pose(position=Point(x=0.75, y=0.0, z=0.0)),
                osp.join(World.MODEL_DIR, 'cafe_table/model.sdf'))
            Gazebo.load_gazebo_model(
                'block',
                Pose(position=Point(x=0.5725, y=0.1265, z=0.90)),
                osp.join(World.MODEL_DIR, 'block/model.urdf'))
            block = Block(
                name='block',
                initial_pos=(0.5725, 0.1265, 0.90),
                random_delta_range=0.15,
                resource=osp.join(World.MODEL_DIR, 'block/model.urdf'))
            # Waiting models to be loaded
            rospy.sleep(1)
            self._block_states_subs.append(
                rospy.Subscriber('/gazebo/model_states', ModelStates,
                                 self._gazebo_update_block_states))
            self._blocks.append(block)
        else:
            for vicon_topic in VICON_TOPICS:
                block = Block(
                    name='block',
                    initial_pos=(0.5725, 0.1265, 0.90),
                    random_delta_range=0.15,
                    resource=vicon_topic)
                self._block_states_subs.append(
                    rospy.Subscriber(block.resource, TransformStamped,
                                     self._vicon_update_block_states))
                self._blocks.append(block)

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

    def _gazebo_update_block_states(self, data):
        model_states = data
        model_names = model_states.name
        for block in self._blocks:
            block_idx = model_names.index(block.name)
            block_pose = model_states.pose[block_idx]
            block.position = block_pose.position
            block.orientation = block_pose.orientation

    def _vicon_update_block_states(self, data):
        translation = data.transform.translation
        rotation = data.transform.rotation
        child_frame_id = data.child_frame_id

        for block in self._blocks:
            if block.resource == child_frame_id:
                block.position = translation
                block.orientation = rotation

    def reset(self):
        if self._simulated:
            self._reset_sim()
        else:
            self._reset_real()

    def _reset_sim(self):
        """
        reset the simulation
        """
        # Randomize start position of blocks
        for block in self._blocks:
            block_random_delta = np.zeros(2)
            while np.linalg.norm(block_random_delta) < 0.1:
                block_random_delta = np.random.uniform(
                    -block.random_delta_range,
                    block.random_delta_range,
                    size=2)
            Gazebo.set_model_pose(
                block.name,
                new_pose=Pose(
                    position=Point(
                        x=block.initial_pos.x + block_random_delta[0],
                        y=block.initial_pos.y + block_random_delta[1],
                        z=block.initial_pos.z)))

    def _reset_real(self):
        """
        reset the real
        """
        # randomize start position of blocks
        for block in self._blocks:
            block_random_delta = np.zeros(2)
            new_pos = block.initial_pos
            while np.linalg.norm(block_random_delta) < 0.1:
                block_random_delta = np.random.uniform(
                    -block.random_delta_range,
                    block.random_delta_range,
                    size=2)
            new_pos.x += block_random_delta[0]
            new_pos.y += block_random_delta[1]
            logger.log('new position for {} is x = {}, y = {}, z = {}'.format(
                block.name, new_pos.x, new_pos.y, new_pos.z))
            ready = False
            while not ready:
                ans = input(
                    'Have you finished setting up {}?[Yes/No]\n'.format(
                        block.name))
                if ans.lower() == 'yes' or ans.lower() == 'y':
                    ready = True

    def terminate(self):
        for sub in self._block_states_subs:
            sub.unregister()

        if self._simulated:
            for block in self._blocks:
                Gazebo.delete_gazebo_model(block.name)
            Gazebo.delete_gazebo_model('table')
        else:
            ready = False
            while not ready:
                ans = input('Are you ready to exit?[Yes/No]\n')
                if ans.lower() == 'yes' or ans.lower() == 'y':
                    ready = True

        self._moveit_scene.remove_world_object('table')

    def get_observation(self):
        blocks_pos = np.array([])
        blocks_ori = np.array([])

        for block in self._blocks:
            pos = np.array(
                [block.position.x, block.position.y, block.position.z])
            ori = np.array([
                block.orientation.x, block.orientation.y, block.orientation.z,
                block.orientation.w
            ])
            blocks_pos = np.concatenate((blocks_pos, pos))
            blocks_ori = np.concatenate((blocks_ori, ori))

        achieved_goal = np.squeeze(blocks_pos)

        obs = np.concatenate((blocks_pos, blocks_ori))

        Observation = collections.namedtuple('Observation',
                                             'obs achieved_goal')

        observation = Observation(obs=obs, achieved_goal=achieved_goal)

        return observation

    @property
    def observation_space(self):
        return gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.get_observation().obs.shape,
            dtype=np.float32)

    def add_block(self, block):
        if self._simulated:
            Gazebo.load_gazebo_model(
                block.name, Pose(position=block.initial_pos), block.resource)
            # Waiting model to be loaded
            rospy.sleep(1)
        else:
            self._block_states_subs.append(
                rospy.Subscriber(block.resource, TransformStamped,
                                 self._vicon_update_block_states))
        self._blocks.append(block)
