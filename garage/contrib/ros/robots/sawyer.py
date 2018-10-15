"""Sawyer Interface."""

from geometry_msgs.msg import Pose
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
                 has_gripper=True,
                 control_mode='position',
                 tip_name='right_hand'):
        """
        Sawyer class.

        :param initial_joint_pos: {str: float}
                        {'joint_name': position_value}, and also
                        initial_joint_pos should include all of the
                        joints that user wants to control and observe.
        :param moveit_group: str
                        Use this to check safety
        :param has_gripper: Bool
                        If use gripper
        :param control_mode: string
                        robot control mode: 'position' or velocity
                        or effort
        :param tip_name: string
                        tip name
        """
        Robot.__init__(self)
        self._limb = intera_interface.Limb('right')
        if not has_gripper:
            self._gripper = intera_interface.Gripper()
        self._initial_joint_pos = initial_joint_pos
        self.control_mode = control_mode
        self._has_gripper = has_gripper
        self._used_joints = []
        self._tip_name = tip_name
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
        if self._has_gripper:
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
        endpoint_pos = np.array(self._limb.endpoint_pose()['position'])
        endpoint_ori = np.array(self._limb.endpoint_pose()['orientation'])
        endpoint_lvel = np.array(self._limb.endpoint_velocity()['linear'])
        endpoint_avel = np.array(self._limb.endpoint_velocity()['angular'])
        endpoint_force = np.array(self._limb.endpoint_effort()['force'])
        endpoint_torque = np.array(self._limb.endpoint_effort()['torque'])

        # joint space
        robot_joint_angles = np.array(list(self._limb.joint_angles().values()))
        robot_joint_velocities = np.array(
            list(self._limb.joint_velocities().values()))
        robot_joint_efforts = np.array(
            list(self._limb.joint_efforts().values()))

        obs = np.concatenate(
            (endpoint_pos, endpoint_ori, endpoint_lvel, endpoint_avel,
             endpoint_force, endpoint_torque, robot_joint_angles,
             robot_joint_velocities, robot_joint_efforts))
        if self._has_gripper:
            obs = np.concatenate(
                (obs, np.array([float(self._gripper.is_gripping())])))
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

    def ik_move(self, desired_pose):
        """
        Use it in task space control.

        :param desired_pose: Pose
                desired gripper pose
        """
        joint_angles = self._limb.ik_request(desired_pose, self._tip_name)
        if joint_angles and self.safety_predict(joint_angles):
            self._limb.set_joint_positions(joint_angles)
        else:
            rospy.logerr(
                "No Joint Angles provided for move_to_joint_positions. Staying put."
            )

    def send_command(self, commands):
        """
        Send command to sawyer.

        :param commands: [float]
                    list of command for different joints and gripper
        """
        action_space = self.action_space
        commands = np.clip(commands, action_space.low, action_space.high)

        if self.control_mode == 'task_space':
            desired_pose = Pose()
            current_pose = self.endpoint_pose
            desired_pose.orientation.w = current_pose['orientation'].w
            desired_pose.orientation.x = current_pose['orientation'].x
            desired_pose.orientation.y = current_pose['orientation'].y
            desired_pose.orientation.z = current_pose['orientation'].z
            desired_pose.position.x = current_pose['position'].x + commands[0]
            desired_pose.position.y = current_pose['position'].y + commands[1]
            desired_pose.position.z = current_pose['position'].z + commands[2]
            self.ik_move(desired_pose)
        else:
            i = 0
            joint_commands = {}
            for joint in self._used_joints:
                joint_commands[joint] = commands[i]
                i += 1

            if self.control_mode == 'position':
                self._set_limb_joint_positions(joint_commands)
            elif self.control_mode == 'velocity':
                self._set_limb_joint_velocities(joint_commands)
            elif self.control_mode == 'effort':
                self._set_limb_joint_torques(joint_commands)

        if self._has_gripper:
            if self.control_mode == 'task_space':
                idx = 4
            else:
                idx = 7
            if commands[idx] > 50:
                self._gripper.open()
                rospy.sleep(0.5)
            else:
                self._gripper.close()
                rospy.sleep(0.5)

    @property
    def endpoint_pose(self):
        """
        Get the endpoint pose.

        :return: endpoint pose
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

        if self.control_mode == 'task_space':
            lower_bounds = np.repeat(-0.03, 3)
            upper_bounds = np.repeat(0.03, 3)
        else:
            for joint in self._used_joints:
                joint_idx = self._joint_limits.joint_names.index(joint)
                if self.control_mode == 'position':
                    lower_bounds = np.concatenate(
                        (lower_bounds,
                         np.array(self._joint_limits.position_lower[
                             joint_idx:joint_idx + 1])))
                    upper_bounds = np.concatenate(
                        (upper_bounds,
                         np.array(self._joint_limits.position_upper[
                             joint_idx:joint_idx + 1])))
                elif self.control_mode == 'velocity':
                    velocity_limit = np.array(
                        self._joint_limits.velocity[joint_idx:joint_idx +
                                                    1]) * 0.1
                    lower_bounds = np.concatenate((lower_bounds,
                                                   -velocity_limit))
                    upper_bounds = np.concatenate((upper_bounds,
                                                   velocity_limit))
                elif self.control_mode == 'effort':
                    effort_limit = np.array(
                        self._joint_limits.effort[joint_idx:joint_idx + 1])
                    lower_bounds = np.concatenate((lower_bounds,
                                                   -effort_limit))
                    upper_bounds = np.concatenate((upper_bounds, effort_limit))
                else:
                    raise ValueError(
                        'Control mode %s is not known!' % self.control_mode)

        if not self._has_gripper:
            return gym.spaces.Box(lower_bounds, upper_bounds, dtype=np.float32)
        else:
            return gym.spaces.Box(
                np.concatenate((lower_bounds, np.array([0]))),
                np.concatenate((upper_bounds, np.array([100]))),
                dtype=np.float32)
