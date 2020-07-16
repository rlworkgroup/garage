#!/usr/bin/env python3
"""Simulates pre-learned policy."""
import argparse
import sys

import cloudpickle
import tensorflow as tf

from garage.sampler.utils import rollout

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from garage.torch import set_gpu_mode



def plot(rewards, tag):
    
    fig, ax = plt.subplots(1, 1, figsize=(6.75, 4))

    reward_df = pd.DataFrame(rewards).melt()
    ax = sns.lineplot(x='variable', y='value', data=reward_df, ax=ax, ci=95, lw=.5)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('obj_to_target')
    ax.set_title('Avg obj_to_target')

    plt.subplots_adjust(top=.85)
    fig.suptitle(f'{tag} (n={rewards.shape[0]})')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if not os.path.exists('figures'):
        os.mkdir('figures')
    fig.savefig(f'figures/{tag}_obj_to_target.jpg')



if __name__ == '__main__':
    set_gpu_mode(True, 0)
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--max_episode_length',
                        type=int,
                        default=500,
                        help='Max length of episode')
    parser.add_argument('--speedup', type=float, default=1, help='Speedup')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.compat.v1.Session():
    #     [rest of the code]
    with tf.compat.v1.Session() as sess:
        with open(args.file, 'rb') as fi:
            data = cloudpickle.load(fi)
        policy = data['algo'].policy
        env = data['env']
        rewards = []
        for i in range(100):
            path = rollout(env,
                           policy,
                           max_episode_length=args.max_episode_length,
                           animated=False,
                           speedup=args.speedup)
            rewards.append(path['env_infos']['obj_to_target'])

    rewards = np.array(rewards)
    plot(rewards, "SAC-pick-place-v2-1_4m_timesteps")

            

