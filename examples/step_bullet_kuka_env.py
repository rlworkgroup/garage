#!/usr/bin/env python3
"""Example of how to load, step, and visualize a Bullet Kuka environment.

This example requires that garage[bullet] be installed.

Note that pybullet_envs is imported so that bullet environments are
registered in gym registry.
"""
# yapf: disable

import click
import gym
import pybullet_envs  # noqa: F401  # pylint: disable=unused-import

from garage.envs import GymEnv

# yapf: enable


@click.command()
@click.option('--n_steps',
              default=1000,
              type=int,
              help='Number of steps to run')
def step_bullet_kuka_env(n_steps=1000):
    """Load, step, and visualize a Bullet Kuka environment.

    Args:
        n_steps (int): number of steps to run.

    """
    # Construct the environment
    env = GymEnv(gym.make('KukaBulletEnv-v0', renders=True, isDiscrete=True))

    # Reset the environment and launch the viewer
    env.reset()
    env.visualize()

    step_count = 0
    es = env.step(env.action_space.sample())
    while not es.last and step_count < n_steps:
        es = env.step(env.action_space.sample())
        step_count += 1


step_bullet_kuka_env()
