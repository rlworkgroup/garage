import argparse
import json
import os.path as osp
import sys
import time

import joblib
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.spatial import ConvexHull
import tensorflow as tf

from sandbox.embed2learn.envs.util import colormap_mpl


def rollout(env,
            agent,
            z,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False,
            goal_markers=None):

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
    agent.reset()

    # Sample embedding network
    # NOTE: it is important to do this _once per rollout_, not once per
    # timestep, since we need correlated noise.
    # t = env.active_task_one_hot
    # z, latent_info = agent.get_latent(t)

    if animated:
        env.render()

    path_length = 0
    while path_length < max_path_length:
        a, agent_info = agent.get_action_from_latent(z, o)
        next_o, r, d, env_info = env.step(a)
        observations.append(agent.observation_space.flatten(o))
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            for i, g in enumerate(goal_markers):
                env.env.get_viewer().add_marker(
                        pos=g,
                        size=0.01 * np.ones(3),
                        label="Task {}".format(i + 1)
                )
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return np.array(observations)


def rollout_interpolate(env,
                        agent,
                        zs,
                        z_steps=10,
                        z_path_length=20,
                        max_path_length=np.inf,
                        animated=False,
                        speedup=1,
                        always_return_paths=False,
                        goal_markers=None):

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
    agent.reset()

    if animated:
        env.render()

    # Do full rollout to first task
    path_length = 0
    while path_length < max_path_length:
        z = zs[0]
        a, agent_info = agent.get_action_from_latent(z, o)
        next_o, r, d, env_info = env.step(a)
        observations.append(agent.observation_space.flatten(o))
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            for i, g in enumerate(goal_markers):
                env.env.get_viewer().add_marker(
                        pos=g,
                        size=0.01 * np.ones(3),
                        label="Task {}".format(i + 1)
                )
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)

    # Interpolate rollout to next task
    for x in np.linspace(0, 1, z_steps):
        z = x * zs[0] + (1 - x) * zs[1]
        for _ in range(z_path_length):
            a, agent_info = agent.get_action_from_latent(z, o)
            next_o, r, d, env_info = env.step(a)
            observations.append(agent.observation_space.flatten(o))
            path_length += 1
            if d:
                break
            o = next_o
            if animated:
                for i, g in enumerate(goal_markers):
                    env.env.get_viewer().add_marker(
                            pos=g,
                            size=0.01 * np.ones(3),
                            label="Task {}".format(i + 1)
                    )
                env.render()
                timestep = 0.05
                time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return np.array(observations)


def get_z_dist(t, policy):
    """ Get the latent distribution for a task """
    onehot = np.zeros(policy.task_space.shape, dtype=np.float32)
    onehot[t] = 1
    _, latent_info = policy.get_latent(onehot)
    return latent_info["mean"], np.exp(latent_info["log_std"])


def play(pkl_file):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Unpack the snapshot
        snapshot = joblib.load(pkl_file)
        env = snapshot["env"]
        policy = snapshot["policy"]

        # Tasks and goals
        num_tasks = policy.task_space.flat_dim
        task_envs = env.env._task_envs
        goals = np.array(
            [te.env._goal_configuration.object_pos for te in task_envs])
        task_cmap = colormap_mpl(num_tasks)

        # Embedding distributions
        z_dists = [get_z_dist(t, policy) for t in range(num_tasks)]
        z_means = np.array([d[0] for d in z_dists])
        z_stds = np.array([d[1] for d in z_dists])

        # Render individual task policies
        for t in range(num_tasks):
            z = z_means[t]

            # Run rollout
            print("Animating task {}".format(t + 1))
            rollout(
                task_envs[0],
                policy,
                z,
                max_path_length=500,
                animated=True,
                goal_markers=goals,
            )

        while True:
            for t in range(num_tasks - 1):
                print("Rollout policy given mean embedding of tasks {} and {}".format(t+1, t+2))
                z = get_mean_embedding2(z_means[t], z_means[t+1], .5)
                rollout(
                    task_envs[0],
                    policy,
                    z,
                    max_path_length=500,
                    animated=True,
                    goal_markers=goals,
                )

            # print("Rollout policy given mean embedding of tasks {} and {}".format(1, 3))
            # z = (z_means[0] + z_means[2]) / 2
            # rollout(
            #     task_envs[0],
            #     policy,
            #     z,
            #     max_path_length=500,
            #     animated=True,
            #     goal_markers=goals,
            # )
            #
            # print("Rollout policy given mean embedding of tasks {} and {}".format(2, 4))
            # z = (z_means[1] + z_means[3]) / 2
            # rollout(
            #     task_envs[0],
            #     policy,
            #     z,
            #     max_path_length=500,
            #     animated=True,
            #     goal_markers=goals,
            # )
            #
            # print("Rollout policy given mean embedding of tasks {} and {}".format(1, 4))
            # z = (z_means[0] + z_means[3]) / 2
            # rollout(
            #     task_envs[0],
            #     policy,
            #     z,
            #     max_path_length=500,
            #     animated=True,
            #     goal_markers=goals,
            # )


def get_mean_embedding1(z1, z2, alpha):
    return np.power(z1, alpha) * np.power(z2, 1. - alpha)


def get_mean_embedding2(z1, z2, alpha):
    return z1 * alpha + z2 * (1 - alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument('pkl_file', metavar='pkl_file', type=str,
                    help='.pkl file containing the policy')
    args = parser.parse_args()

    play(args.pkl_file)
