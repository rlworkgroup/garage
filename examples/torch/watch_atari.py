#!/usr/bin/env python3
"""Utility to watch a trained agent play an Atari game."""

import click
import gym
import numpy as np

from garage import rollout
from garage.envs import GymEnv
from garage.envs.wrappers.clip_reward import ClipReward
from garage.envs.wrappers.episodic_life import EpisodicLife
from garage.envs.wrappers.fire_reset import FireReset
from garage.envs.wrappers.grayscale import Grayscale
from garage.envs.wrappers.max_and_skip import MaxAndSkip
from garage.envs.wrappers.noop import Noop
from garage.envs.wrappers.resize import Resize
from garage.envs.wrappers.stack_frames import StackFrames
from garage.experiment import Snapshotter


# pylint: disable=no-value-for-parameter, protected-access
@click.command()
@click.argument('saved_dir', type=str)
@click.option('--env', type=str, default=None)
@click.option('--num_episodes', type=int, default=10)
def watch_atari(saved_dir, env=None, num_episodes=10):
    """Watch a trained agent play an atari game.

    Args:
        saved_dir (str): Directory containing the pickle file.
        env (str): Environment to run episodes on. If None, the pickled
            environment is used.
        num_episodes (int): Number of episodes to play. Note that when using
            the EpisodicLife wrapper, an episode is considered done when a
            life is lost. Defaults to 10.
    """
    snapshotter = Snapshotter()
    data = snapshotter.load(saved_dir)
    if env is not None:
        env = gym.make(env)
        env = Noop(env, noop_max=30)
        env = MaxAndSkip(env, skip=4)
        env = EpisodicLife(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireReset(env)
        env = Grayscale(env)
        env = Resize(env, 84, 84)
        env = ClipReward(env)
        env = StackFrames(env, 4, axis=0)
        env = GymEnv(env)
    else:
        env = data['env']

    exploration_policy = data['algo'].exploration_policy
    exploration_policy.policy._qf.to('cpu')
    ep_rewards = np.asarray([])
    for _ in range(num_episodes):
        episode_data = rollout(env,
                               exploration_policy.policy,
                               animated=True,
                               pause_per_frame=0.02)
        ep_rewards = np.append(ep_rewards, np.sum(episode_data['rewards']))

    print('Average Reward {}'.format(np.mean(ep_rewards)))


watch_atari()
