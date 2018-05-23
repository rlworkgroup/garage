from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelState
import rospy


class Gazebo(object):
    def __init__(self):
        """
        Gazebo Service Util
        """
        self._set_model_pose_pub = rospy.Publisher(
            '/gazebo/set_model_state', ModelState, queue_size=20)
        self._spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model',
                                             SpawnModel)
        self._spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model',
                                              SpawnModel)
        self._delete_model = rospy.ServiceProxy('/gazebo/delete_model',
                                                DeleteModel)

    def set_model_pose(self, model_name, new_pose, reference_frame):
        """
        Set the gazebo model's pose
        :param model_name: str
                    the name of new model
        :param new_pose: geometry_msgs.msg.Pose
                    the pose of new model
        :param reference_frame: str
                    the reference frame name(e.g. 'world')
        """
        msg = ModelState()
        msg.model_name = model_name
        msg.pose = new_pose
        msg.reference_frame = reference_frame
        self._set_model_pose_pub.publish(msg)

    def load_gazebo_sdf_model(self,
                              model_name,
                              model_pose,
                              model_path,
                              model_reference_frame='world'):
        # Load target SDF
        with open(model_path, 'r') as model_file:
            model_xml = model_file.read()
        # Spawn Target SDF
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        self._spawn_sdf(model_name, model_xml, '/', model_pose,
                        model_reference_frame)

    def load_gazebo_urdf_model(self,
                               model_name,
                               model_pose,
                               model_path,
                               model_reference_frame='world'):
        # Load target URDF
        with open(model_path, 'r') as model_file:
            model_xml = model_file.read()
        # Spawn Target URDF
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        self._spawn_urdf(model_name, model_xml, '/', model_pose,
                         model_reference_frame)

    def delete_gazebo_model(self, model_name):
        rospy.wait_for_service('/gazebo/delete_model')
        self._delete_model(model_name)
