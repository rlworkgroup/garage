"""
Push task for the sawyer robot
"""

import rospy

from rllab.core.serializable import Serializable

from contrib.ros.envs import sawyer_env

INITIAL_MODEL_POS = {
    'table0': [0.75, 0.0, 0.0],
    'object0': [0.5725, 0.1265, 0.80]
}

INITIAL_ROBOT_JOINT_POS = {
    'right_j0': -0.041662954890248294,
    'right_j1': -1.0258291091425074,
    'right_j2': 0.0293680414401436,
    'right_j3': 2.17518162913313,
    'right_j4': -0.06703022873354225,
    'right_j5': 0.3968371433926965,
    'right_j6': 1.7659649178699421,
}


class PushEnv(sawyer_env.SawyerEnv, Serializable):
    def __init__(self, sparse_reward=False):
        Serializable.quick_init(self, locals())

        sawyer_env.SawyerEnv.__init__(
            self,
            initial_robot_joint_pos=INITIAL_ROBOT_JOINT_POS,
            robot_control_mode='effort',
            has_object=True,
            distance_threshold=0.05,
            initial_model_pos=INITIAL_MODEL_POS,
            sparse_reward=sparse_reward,
            target_in_the_air=False,
            simulated=True,
            target_range=0.15,
            obj_range=0.15)