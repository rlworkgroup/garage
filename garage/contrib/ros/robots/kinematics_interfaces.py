"""Interfaces to MoveIt! kinematics services."""

from geometry_msgs.msg import PoseStamped
import moveit_msgs.srv
import rospy
from sensor_msgs.msg import JointState


class ForwardKinematics:
    """Interface to MoveIt! forward kinematics service."""

    def __init__(self):
        """Interface to MoveIt! forward kinematics service."""
        self._srv = rospy.ServiceProxy('/compute_fk',
                                       moveit_msgs.srv.GetPositionFK)
        self._srv.wait_for_service()

    def terminate(self):
        """Terminate the service proxy."""
        self._srv.close()

    def get_fk(self,
               fk_link_names,
               joint_names,
               positions,
               frame_id='base_link'):
        """
        Get the forward kinematics of a joint configuration.

        :param fk_link_names: [string]
                        list of links that we want to
                        get the forward kinematics from
        :param joint_names: [string]
                        with the joint names to set a
                        position to ask for the FK
        :param positions: [float]
                        positions of joints
        :param frame_id: string
                        the reference frame to be used
        :return fk_result: moveit.srv.GetPositionFKResponse
        """
        fk_request = moveit_msgs.srv.GetPositionFKRequest()
        fk_request.fk_link_names = fk_link_names
        fk_request.robot_state.joint_state = joint_names
        fk_request.robot_state.joint_state.position = positions
        fk_request.header.frame_id = frame_id

        fk_result = self._srv.call(fk_request)
        return fk_result

    def get_current_fk(self, fk_link_names, frame_id='base_link'):
        """
        Get the current forward kinematics of a set of links.

        :param fk_link_names: [string]
                        list of links that we want to
                        get the forward kinematics from
        :param frame_id: string
                        the reference frame to be used
        :return fk_result: moveit_msgs.srv.GetPositionFKResponse
        """
        # Subscribe to a joint_states
        js = rospy.wait_for_message('/robot/joint_states', JointState)
        # Call FK service
        fk_result = self.get_fk(fk_link_names, js.name, js.position, frame_id)
        return fk_result


class InverseKinematics:
    """Interface to MoveIt! inverse kinematics service."""

    def __init__(self):
        """Interface to MoveIt! inverse kinematics service."""
        self._srv = rospy.ServiceProxy('/compute_ik',
                                       moveit_msgs.srv.GetPositionIK)
        self._srv.wait_for_service()

    def terminate(self):
        """Terminate."""
        self._srv.close()

    def get_ik(self,
               group_name,
               ik_link_name,
               pose_stamped,
               avoid_collisions=True,
               attempts=None,
               robot_state=None,
               constraints=None):
        """
        Get the inverse kinematics with a link in a pose in 3d world.

        :param group_name: string
                    group name, i.e. 'right_arm'
        :param ik_link_name: string
                    link that will be in the pose given to evaluate the IK
        :param pose_stamped: PoseStamped
                    the pose with frame_id of the link
        :param avoid_collisions: Bool
                    if we want solutions with collision avoidance
        :param attempts: Int
                    number of attempts to get an IK
        :param robot_state: RobotState
                    the robot state where to start searching IK from
                    (optional, current pose will be used if ignored)
        :param constraints:
        :return ik_result: moveit_msgs.srv.GetPositionIKResponse
        """
        assert isinstance(pose_stamped, type(PoseStamped))

        ik_request = moveit_msgs.srv.GetPositionIKRequest()
        ik_request.ik_request.group_name = group_name
        if robot_state:
            ik_request.ik_request.robot_state = robot_state
        ik_request.ik_request.avoid_collisions = avoid_collisions
        ik_request.ik_request.ik_link_name = ik_link_name
        ik_request.ik_request.pose_stamped = pose_stamped
        if attempts:
            ik_request.ik_request.attempts = attempts
        else:
            ik_request.ik_request.attempts = 1
        if constraints:
            ik_request.ik_request.constraints = constraints

        ik_result = self._srv.call(ik_request)
        return ik_result


class StateValidity:
    """Interface to MoveIt! StateValidity service."""

    def __init__(self):
        """Interface to MoveIt! StateValidity service."""
        self._srv = rospy.ServiceProxy('/check_state_validity',
                                       moveit_msgs.srv.GetStateValidity)
        self._srv.wait_for_service()

    def terminate(self):
        """Terminate."""
        self._srv.close()

    def get_state_validity(self,
                           robot_state,
                           group_name='right_arm',
                           constraints=None):
        """
        Get state validity.

        :param robot_state: RobotState
                        robot state
        :param group_name: string
                        planner group name
        :param constraints:
        :return result:
        """
        sv_request = moveit_msgs.srv.GetStateValidityRequest()
        sv_request.robot_state = robot_state
        sv_request.group_name = group_name
        if constraints:
            sv_request.constraints = constraints
        result = self._srv.call(sv_request)
        return result
