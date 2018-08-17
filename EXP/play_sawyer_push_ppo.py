import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import time

import joblib
import numpy as np
import tensorflow as tf

from garage.misc import tensor_utils


def rollout(env,
            agent,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        # a = agent_info["mean"]
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
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return None

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )


def play(pkl_file):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Unpack the snapshot
        snapshot = joblib.load(pkl_file)
        env = snapshot["env"]
        policy = snapshot["policy"]

        while True:
            rollout(
                env,
                policy,
                max_path_length=500,
                animated=True,
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
