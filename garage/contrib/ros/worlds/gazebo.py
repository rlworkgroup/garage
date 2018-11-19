import os.path as osp

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import DeleteModel, SpawnModel
import rospy


class Gazebo:
    set_model_pose_pub = rospy.Publisher(
        '/gazebo/set_model_state', ModelState, queue_size=20)
    spawn_sdf = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
    spawn_urdf = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
    delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

    @classmethod
    def set_model_pose(cls,
                       model_name,
                       new_pose,
                       model_reference_frame='world'):
        """
        Set the gazebo model's pose
        :param model_name: str
                    the name of new model
        :param new_pose: geometry_msgs.msg.Pose
                    the pose of new model
        :param model_reference_frame: str
                    the reference frame name(e.g. 'world')
        """
        msg = ModelState()
        msg.model_name = model_name
        msg.pose = new_pose
        msg.reference_frame = model_reference_frame
        cls.set_model_pose_pub.publish(msg)

    @classmethod
    def _load_gazebo_sdf_model(cls,
                               model_name,
                               model_pose,
                               model_path,
                               model_reference_frame='world'):
        # Load target SDF
        with open(model_path, 'r') as model_file:
            model_xml = model_file.read()
        # Spawn Target SDF
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        cls.spawn_sdf(model_name, model_xml, '/', model_pose,
                      model_reference_frame)

    @classmethod
    def _load_gazebo_urdf_model(cls,
                                model_name,
                                model_pose,
                                model_path,
                                model_reference_frame='world'):
        # Load target URDF
        with open(model_path, 'r') as model_file:
            model_xml = model_file.read()
        # Spawn Target URDF
        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        cls.spawn_urdf(model_name, model_xml, '/', model_pose,
                       model_reference_frame)

    @classmethod
    def delete_gazebo_model(cls, model_name):
        rospy.wait_for_service('/gazebo/delete_model')
        cls.delete_model(model_name)

    @classmethod
    def load_gazebo_model(cls, name, model_pose, model_path):
        if cls._is_sdf(model_path):
            cls._load_gazebo_sdf_model(name, model_pose, model_path)
        else:
            cls._load_gazebo_urdf_model(name, model_pose, model_path)

    @classmethod
    def _is_sdf(cls, file_path):
        """

        :param file_path: str
                the model file path
        :return _is_sdf: bool
            True: is sdf model
        """
        file_ext = osp.splitext(file_path)[-1]
        return file_ext == '.sdf'
