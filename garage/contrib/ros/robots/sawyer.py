"""Sawyer Interface."""

import gym
from intera_core_msgs.msg import JointLimits
import intera_interface
import moveit_msgs.msg
import numpy as np
import rospy

from garage.contrib.ros.robots.kinematics_interfaces import StateValidity
from garage.contrib.ros.robots.robot import Robot


class Sawyer(Robot):
    """Sawyer class."""

    def __init__(self,
                 initial_joint_pos,
                 moveit_group,
                 control_mode='position'):
        """
        Sawyer class.

        :param initial_joint_pos: {str: float}
                        {'joint_name': position_value}, and also
                        initial_joint_pos should include all of the
                        joints that user wants to control and observe.
        :param moveit_group: str
                        Use this to check safety
        :param control_mode: string
                        robot control mode: 'position' or velocity
                        or effort
        """
        Robot.__init__(self)
        self._limb = intera_interface.Limb('right')
        self._gripper = intera_interface.Gripper()
        self._initial_joint_pos = initial_joint_pos
        self._control_mode = control_mode
        self._used_joints = []
        for joint in initial_joint_pos:
            self._used_joints.append(joint)
        self._joint_limits = rospy.wait_for_message('/robot/joint_limits',
                                                    JointLimits)
        self._moveit_group = moveit_group

        self._sv = StateValidity()

    def safety_check(self):
        """
        If robot is in safe state.

        :return safe: Bool
                if robot is safe.
        """
        rs = moveit_msgs.msg.RobotState()
        current_joint_angles = self._limb.joint_angles()
        for joint in current_joint_angles:
            rs.joint_state.name.append(joint)
            rs.joint_state.position.append(current_joint_angles[joint])
        result = self._sv.get_state_validity(rs, self._moveit_group)
        return result.valid

    def safety_predict(self, joint_angles):
        """
        Will robot be in safe state.

        :param joint_angles: {'': float}
        :return safe: Bool
                    if robot is safe.
        """
        rs = moveit_msgs.msg.RobotState()
        for joint in joint_angles:
            rs.joint_state.name.append(joint)
            rs.joint_state.position.append(joint_angles[joint])
        result = self._sv.get_state_validity(rs, self._moveit_group)
        return result.valid

    @property
    def enabled(self):
        """
        If robot is enabled.

        :return: if robot is enabled.
        """
        return intera_interface.RobotEnable(
            intera_interface.CHECK_VERSION).state().enabled

    def _set_limb_joint_positions(self, joint_angle_cmds):
        # limit joint angles cmd
        current_joint_angles = self._limb.joint_angles()
        for joint in joint_angle_cmds:
            joint_cmd_delta = joint_angle_cmds[joint] - \
                              current_joint_angles[joint]
            joint_angle_cmds[
                joint] = current_joint_angles[joint] + joint_cmd_delta * 0.1

        if self.safety_predict(joint_angle_cmds):
            self._limb.set_joint_positions(joint_angle_cmds)

    def _set_limb_joint_velocities(self, joint_angle_cmds):
        self._limb.set_joint_velocities(joint_angle_cmds)

    def _set_limb_joint_torques(self, joint_angle_cmds):
        self._limb.set_joint_torques(joint_angle_cmds)

    def _set_gripper_position(self, position):
        self._gripper.set_position(position)

    def _move_to_start_position(self):
        if rospy.is_shutdown():
            return
        self._limb.move_to_joint_positions(
            self._initial_joint_pos, timeout=5.0)
        self._gripper.open()
        rospy.sleep(1.0)

    def reset(self):
        """Reset sawyer."""
        self._move_to_start_position()

    def get_observation(self):
        """
        Get robot observation.

        :return: robot observation
        """
        # cartesian space
        gripper_pos = np.array(self._limb.endpoint_pose()['position'])
        gripper_ori = np.array(self._limb.endpoint_pose()['orientation'])
        gripper_lvel = np.array(self._limb.endpoint_velocity()['linear'])
        gripper_avel = np.array(self._limb.endpoint_velocity()['angular'])
        gripper_force = np.array(self._limb.endpoint_effort()['force'])
        gripper_torque = np.array(self._limb.endpoint_effort()['torque'])

        # joint space
        robot_joint_angles = np.array(list(self._limb.joint_angles().values()))
        robot_joint_velocities = np.array(
            list(self._limb.joint_velocities().values()))
        robot_joint_efforts = np.array(
            list(self._limb.joint_efforts().values()))

        obs = np.concatenate(
            (gripper_pos, gripper_ori, gripper_lvel, gripper_avel,
             gripper_force, gripper_torque, robot_joint_angles,
             robot_joint_velocities, robot_joint_efforts))
        return obs

    @property
    def observation_space(self):
        """
        Observation space.

        :return: gym.spaces
                    observation space
        """
        return gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.get_observation().shape,
            dtype=np.float32)

    def send_command(self, commands):
        """
        Send command to sawyer.

        :param commands: [float]
                    list of command for different joints and gripper
        """
        action_space = self.action_space
        commands = np.clip(commands, action_space.low, action_space.high)
        i = 0
        joint_commands = {}
        for joint in self._used_joints:
            joint_commands[joint] = commands[i]
            i += 1

        if self._control_mode == 'position':
            self._set_limb_joint_positions(joint_commands)
        elif self._control_mode == 'velocity':
            self._set_limb_joint_velocities(joint_commands)
        elif self._control_mode == 'effort':
            self._set_limb_joint_torques(joint_commands)

        self._set_gripper_position(commands[7])

    @property
    def gripper_pose(self):
        """
        Get the gripper pose.

        :return: gripper pose
        """
        return self._limb.endpoint_pose()

    @property
    def action_space(self):
        """
        Return a Space object.

        :return: action space
        """
        lower_bounds = np.array([])
        upper_bounds = np.array([])
        for joint in self._used_joints:
            joint_idx = self._joint_limits.joint_names.index(joint)
            if self._control_mode == 'position':
                lower_bounds = np.concatenate(
                    (lower_bounds,
                     np.array(self._joint_limits.position_lower[
                         joint_idx:joint_idx + 1])))
                upper_bounds = np.concatenate(
                    (upper_bounds,
                     np.array(self._joint_limits.position_upper[
                         joint_idx:joint_idx + 1])))
            elif self._control_mode == 'velocity':
                velocity_limit = np.array(
                    self._joint_limits.velocity[joint_idx:joint_idx + 1]) * 0.1
                lower_bounds = np.concatenate((lower_bounds, -velocity_limit))
                upper_bounds = np.concatenate((upper_bounds, velocity_limit))
            elif self._control_mode == 'effort':
                effort_limit = np.array(
                    self._joint_limits.effort[joint_idx:joint_idx + 1])
                lower_bounds = np.concatenate((lower_bounds, -effort_limit))
                upper_bounds = np.concatenate((upper_bounds, effort_limit))
            else:
                raise ValueError(
                    'Control mode %s is not known!' % self._control_mode)
        return gym.spaces.Box(
            np.concatenate((lower_bounds, np.array([0]))),
            np.concatenate((upper_bounds, np.array([100]))),
            dtype=np.float32)
