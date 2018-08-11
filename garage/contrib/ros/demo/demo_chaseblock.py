"""Ask robot to chase block."""

import copy
import sys

from geometry_msgs.msg import Pose
import moveit_commander
import intera_interface
import rospy

from garage.contrib.ros.worlds import BlockWorld

BLOCK_NAME = 'block_0'

INITIAL_ROBOT_JOINT_POS = {
    'right_j0': -0.140923828125,
    'right_j1': -1.2789248046875,
    'right_j2': -3.043166015625,
    'right_j3': -2.139623046875,
    'right_j4': -0.047607421875,
    'right_j5': -0.7052822265625,
    'right_j6': -1.4102060546875,
}


class ChaseBlock(object):
    def __init__(self, limb='right', hover_distance=0.0, tip_name="right_gripper_tip"):
        self._limb_name = limb
        self._limb = intera_interface.Limb(limb)
        self._limb.set_joint_position_speed(0.3)
        self._tip_name = tip_name
        self._gripper = intera_interface.Gripper()
        self._hover_distance = hover_distance

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)

    def approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self._limb.ik_request(approach, self._tip_name)
        self._guarded_move_to_joint_position(joint_angles)

    def _guarded_move_to_joint_position(self, joint_angles, timeout=60.0):
        if rospy.is_shutdown():
            return
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles, timeout=timeout)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")


def run():
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node('demo_chaseblock', anonymous=True)

    moveit_robot = moveit_commander.RobotCommander()
    moveit_scene = moveit_commander.PlanningSceneInterface()
    moveit_group_name = 'right_arm'
    moveit_group = moveit_commander.MoveGroupCommander(moveit_group_name)
    moveit_scene = moveit_commander.PlanningSceneInterface()
    moveit_frame = moveit_robot.get_planning_frame()

    blockworld = BlockWorld(moveit_scene, moveit_frame, False)

    blockworld.initialize()

    cb = ChaseBlock()

    cb.move_to_start(INITIAL_ROBOT_JOINT_POS)

    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        block_position = blockworld.get_block_position(BLOCK_NAME)

        target_pose = Pose()

        target_pose.position = block_position
        target_pose.position.z += 0.10
        target_pose.orientation.x = 0
        target_pose.orientation.y = 1
        target_pose.orientation.z = 0
        target_pose.orientation.w = 0

        print('Moving to: ', target_pose)

        cb.approach(target_pose)

        r.sleep()

    blockworld.terminate()


if __name__ == '__main__':
    run()
