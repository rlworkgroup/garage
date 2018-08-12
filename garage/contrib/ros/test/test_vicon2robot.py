import sys

import moveit_commander
import rospy

from garage.contrib.ros.worlds import BlockWorld


def run():
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node('test_vicon2robot', anonymous=True)

    moveit_robot = moveit_commander.RobotCommander()
    moveit_scene = moveit_commander.PlanningSceneInterface()
    moveit_group_name = 'right_arm'
    moveit_group = moveit_commander.MoveGroupCommander(moveit_group_name)
    moveit_scene = moveit_commander.PlanningSceneInterface()
    moveit_frame = moveit_robot.get_planning_frame()

    blockworld = BlockWorld(moveit_scene,
                            moveit_frame,
                            False)

    blockworld.initialize()

    r = rospy.Rate(10)

    while not rospy.is_shutdown():
        print('block_positions: ', blockworld.get_blocks_position())
        print('block orientations: ', blockworld.get_blocks_orientation())

        r.sleep()

    blockworld.terminate()


if __name__ == '__main__':
    run()
