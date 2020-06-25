#!/usr/bin/env python3
"""Example of how to load, step, and visualize a Bullet Kuka environment.

This example requires that garage[bullet] be installed.
"""
import click
import gym

from garage.envs.bullet import BulletEnv


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
    env = BulletEnv(
        gym.make('KukaBulletEnv-v0',
                 renders=True,
                 isDiscrete=True,
                 maxSteps=10000000))

    # Reset the environment and launch the viewer
    env.reset()
    env.render()

    # Step randomly until interrupted
    steps = 0
    while steps < n_steps:
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            break
        steps += 1


step_bullet_kuka_env()
