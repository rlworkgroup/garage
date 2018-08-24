import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys

import joblib
import moveit_commander
import numpy as np
import rospy
import tensorflow as tf

from garage.contrib.ros.envs.sawyer import PusherEnv
from garage.misc import tensor_utils
from garage.tf.envs import TfEnv

INITIAL_ROBOT_JOINT_POS = {
    'right_j0': -0.4839443359375,
    'right_j1': -0.991173828125,
    'right_j2': -2.3821015625,
    'right_j3': -1.9510517578125,
    'right_j4': -0.5477119140625,
    'right_j5': -0.816458984375,
    'right_j6': -0.816326171875,
}


def rollout(env,
            agent,
            max_path_length=np.inf,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        # a = agent_info["mean"]
        a = np.concatenate((a, np.array([0.])))
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        # if d:
        #     break
        o = next_o

    if not always_return_paths:
        return None

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )


def play(pkl_file):
    # config = tf.ConfigProto(
    #     device_count={'GPU': 0}
    # )
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node('rollingout_policy_push', anonymous=True)

    initial_goal = np.array([0.70, 0., 0.03])
    push_env = PusherEnv(
        initial_goal=initial_goal,
        initial_joint_pos=INITIAL_ROBOT_JOINT_POS,
        simulated=False,
        robot_control_mode='position',
        action_scale=0.04
    )
    push_env._robot.set_joint_position_speed(0.05)
    rospy.on_shutdown(push_env.shutdown)
    push_env.initialize()

    env = TfEnv(push_env)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Unpack the snapshot
        snapshot = joblib.load(pkl_file)

        policy = snapshot["policy"]

        while True:
            rollout(
                env,
                policy,
                max_path_length=500,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument(
        'pkl_file',
        metavar='pkl_file',
        type=str,
        help='.pkl file containing the policy')
    args = parser.parse_args()

    play(args.pkl_file)
