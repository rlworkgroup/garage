"""
Sawyer Interface
"""

from intera_core_msgs.msg import JointLimits
import intera_interface
import numpy as np
import rospy

from rllab.misc import logger
from rllab.spaces import Box


class Sawyer(object):
    def __init__(self, initial_joint_pos, control_mode):
        """
        :param initial_joint_pos: {str: float}
                            {'joint_name': position_value}
        """
        self._limb = intera_interface.Limb('right')
        self._gripper = intera_interface.Gripper()
        self._joint_limits = rospy.wait_for_message('/robot/joint_limits',
                                                    JointLimits)
        self._initial_joint_pos = initial_joint_pos
        self._control_mode = control_mode

    @property
    def enabled(self):
        return intera_interface.RobotEnable(
            intera_interface.CHECK_VERSION).state().enabled

    def set_limb_joint_positions(self, joint_angle_cmds):
        self._limb.set_joint_positions(joint_angle_cmds)

    def set_limb_joint_velocities(self, joint_angle_cmds):
        self._limb.set_joint_velocities(joint_angle_cmds)

    def set_limb_joint_torques(self, joint_angle_cmds):
        self._limb.set_joint_torques(joint_angle_cmds)

    def set_gripper_position(self, position):
        self._gripper.set_position(position)

    def move_to_start_position(self):
        if rospy.is_shutdown():
            return
        self._limb.move_to_joint_positions(
            self._initial_joint_pos, timeout=5.0)
        self._gripper.open()
        rospy.sleep(1.0)

    def get_obs(self):
        # gripper's information
        gripper_pos = np.array(self._limb.endpoint_pose()['position'])
        gripper_ori = np.array(self._limb.endpoint_pose()['orientation'])
        gripper_lvel = np.array(self._limb.endpoint_velocity()['linear'])
        gripper_avel = np.array(self._limb.endpoint_velocity()['angular'])
        gripper_force = np.array(self._limb.endpoint_effort()['force'])
        gripper_torque = np.array(self._limb.endpoint_effort()['torque'])

        # robot joints angles
        robot_joint_angles = np.array(list(self._limb.joint_angles().values()))

        obs = np.concatenate(
            (gripper_pos, gripper_ori, gripper_lvel, gripper_avel,
             gripper_force, gripper_torque, robot_joint_angles))
        return obs

    def set_command(self, commands):
        """
        :param commands: [float]
                    list of command for different joints and gripper
        """
        joint_commands = {
            'right_j0': commands[0],
            'right_j1': commands[1],
            'right_j2': commands[2],
            'right_j3': commands[3],
            'right_j4': commands[4],
            'right_j5': commands[5],
            'right_j6': commands[6]
        }
        if self._control_mode == 'position':
            self.set_limb_joint_positions(joint_commands)
        elif self._control_mode == 'velocity':
            self.set_limb_joint_velocities(joint_commands)
        elif self._control_mode == 'effort':
            self.set_limb_joint_torques(joint_commands)

        self.set_gripper_position(commands[7])

    @property
    def gripper_pose(self):
        return self._limb.endpoint_pose()

    @property
    def action_space(self):
        """
        Returns a Space object
        """
        # For sawyer's joints
        # [head_pan, right_gripper_l_finger_joint, right_gripper_r_finger_joint, right_j0, right_j1,
        # right_j2, right_j3, right_j4, right_j5, right_j6]
        # For our use
        # joints[3:10]
        # set the action space depending on different control modes
        if self._control_mode == 'position':
            lower_bounds = np.array(self._joint_limits.position_lower[3:10])
            upper_bounds = np.array(self._joint_limits.position_upper[3:10])
        elif self._control_mode == 'velocity':
            lower_bounds = np.zeros_like(self._joint_limits.velocity[3:10])
            upper_bounds = np.array(self._joint_limits.velocity[3:10])
        elif self._control_mode == 'effort':
            lower_bounds = np.zeros_like(self._joint_limits.effort[3:10])
            upper_bounds = np.array(self._joint_limits.effort[3:10])
        else:
            raise ValueError(
                'Control mode %s is not known!' % self._control_mode)
        return Box(
            np.concatenate((lower_bounds, np.array([0]))),
            np.concatenate((upper_bounds, np.array([100]))))
