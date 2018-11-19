import argparse
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import time

import joblib
import moveit_commander
import numpy as np
import pickle
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

playback_obs = []
playback_actions = []
playback_tasks = []
playback_env_infos = []
playback_latents = []

def rollout(env,
            sim_env,
            agent,
            z,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False,
            goal_markers=None,
            task=""):

    observations = []
    tasks = []
    latents = []
    latent_infos = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    # Resets
    o = env.reset()
    sim_env.reset()
    agent.reset()

    # Sample embedding network
    # NOTE: it is important to do this _once per rollout_, not once per
    # timestep, since we need correlated noise.
    # t = env.active_task_one_hot
    # z, latent_info = agent.get_latent(t)

    # if animated:
    #     sim_env.render()

    path_length = 0
    while path_length < max_path_length:
        a, agent_info = agent.get_action_from_latent(z, o)
        playback_actions.append(a.copy())
        agent_infos.append(copy.deepcopy(agent_info))
        actions.append(copy.deepcopy(a))
        next_o, r, d, env_info = env.step(a)
        playback_env_infos.append(copy.deepcopy(env_info))
        # sim_env.step(a)
        observations.append(copy.deepcopy(agent.observation_space.flatten(o)))
        playback_obs.append(copy.deepcopy(o))
        playback_tasks.append(task)
        playback_latents.append(z)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            for i, g in enumerate(goal_markers):
                sim_env.env.get_viewer().add_marker(
                        pos=g,
                        size=0.01 * np.ones(3),
                        label="Task {}".format(i + 1)
                )
            sim_env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return {
        'observations': np.array(observations),
        'latent:': z,
        'actions': actions,
        'agent_infos': agent_infos,
        'env_infos': env_infos
    }


def play(pkl_file):
    # config = tf.ConfigProto(
    #     device_count={'GPU': 0}
    # )
    moveit_commander.roscpp_initialize(sys.argv)

    rospy.init_node('rollingout_policy_pusher_embed', anonymous=True)

    initial_goal = np.array([0.70, 0., 0.03])
    push_env = PusherEnv(
        initial_goal=initial_goal,
        initial_joint_pos=INITIAL_ROBOT_JOINT_POS,
        simulated=False,
        robot_control_mode='position',
        action_scale=0.04
    )
    # push_env._robot.set_joint_position_speed(0.05)
    rospy.on_shutdown(push_env.shutdown)
    push_env.initialize()

    env = TfEnv(push_env)

    env.env._robot.set_joint_position_speed(0.1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Unpack the snapshot
        snapshot = joblib.load(pkl_file)

        sim_env = snapshot['env']

        policy = snapshot["policy"]

        # Tasks and goals
        num_tasks = policy.task_space.flat_dim
        task_envs = sim_env.env._task_envs
        goals = np.array(
            [te.env._goal_configuration.object_pos for te in task_envs])

        # Embedding distributions
        z_dists = [get_z_dist(t, policy) for t in range(num_tasks)]
        z_means = np.array([d[0] for d in z_dists])
        z_stds = np.array([d[1] for d in z_dists])

        task_dict = {0: 'up', 1: 'down', 2: 'left', 3: 'right', 4: 'up_left', 5: 'down_left', 6: 'down_right', 7: 'up_right'}

        # Render individual task policies
        # for t in range(num_tasks):
        #     z = z_means[(t + 3) % 4]
        #
        #     # Run rollout
        #     print("Animating task {}".format(t + 1))
        #     info = rollout(
        #         env,
        #         task_envs[0],
        #         policy,
        #         z,
        #         max_path_length=200,
        #         animated=False,
        #         goal_markers=goals,
        #         task=task_dict[t]
        #     )
        #
        #     pickle.dump(info, open('task%i_infos.pkl' % t, "wb"))
        #     print('Rollout data is saved!')
        #     playback_data = np.array(dict(actions=np.array(playback_actions),
        #                                   tasks=np.array(playback_tasks),
        #                                   obs=np.array(playback_obs),
        #                                   env_infos=np.array(playback_env_infos),
        #                                   latents=playback_latents))
        #     np.save("playback_pusher.npy", playback_data)

        # while True:
        #     for t in range(num_tasks - 1):
        #         print("Rollout policy given mean embedding of tasks {} and {}".format(t+1, t+2))
        #         z = get_mean_embedding2(z_means[t], z_means[t+1], .5)
        #         rollout(
        #             env,
        #             policy,
        #             z,
        #             max_path_length=400,
        #             animated=False,
        #         )

        print("Rollout policy given mean embedding of tasks {} and {}".format(1, 4))
        z = (z_means[0] + z_means[3]) / 2
        info = rollout(
            env,
            task_envs[0],
            policy,
            z,
            max_path_length=200,
            animated=False,
            goal_markers=goals,
            task=task_dict[7]
        )

        # pickle.dump(info, open('task%i_infos.pkl' % t, "wb"))
        print('Rollout data is saved!')
        playback_data = np.array(dict(actions=np.array(playback_actions),
                                      tasks=np.array(playback_tasks),
                                      obs=np.array(playback_obs),
                                      env_infos = np.array(playback_env_infos),
                                      latents=playback_latents)
        )
        np.save("playback_pusher.npy", playback_data)

        env.reset()

            # print("Rollout policy given mean embedding of tasks {} and {}".format(2, 4))
            # z = (z_means[1] + z_means[3]) / 2
            # rollout(
            #     env,
            #     policy,
            #     z,
            #     max_path_length=400,
            #     animated=False,
            # )
            #
            # print("Rollout policy given mean embedding of tasks {} and {}".format(1, 4))
            # z = (z_means[0] + z_means[3]) / 2
            # rollout(
            #     env,
            #     policy,
            #     z,
            #     max_path_length=400,
            #     animated=False,
            # )


def get_z_dist(t, policy):
    """ Get the latent distribution for a task """
    onehot = np.zeros(policy.task_space.shape, dtype=np.float32)
    onehot[t] = 1
    _, latent_info = policy.get_latent(onehot)
    return latent_info["mean"], np.exp(latent_info["log_std"])


def get_mean_embedding1(z1, z2, alpha):
    return np.power(z1, alpha) * np.power(z2, 1. - alpha)


def get_mean_embedding2(z1, z2, alpha):
    return z1 * alpha + z2 * (1 - alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument(
        'pkl_file',
        metavar='pkl_file',
        type=str,
        help='.pkl file containing the policy')
    args = parser.parse_args()

    play(args.pkl_file)
