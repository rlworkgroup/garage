#!/usr/bin/env python3
"""Example of how to load, step, and visualize an environment.

This example requires that garage[pybullet] be installed.
"""
import argparse

from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

from garage.envs.pybullet import PybulletEnv

parser = argparse.ArgumentParser()
parser.add_argument('--n_steps',
                    type=int,
                    default=1000,
                    help='Number of steps to run')
args = parser.parse_args()

# Construct the environment
env = PybulletEnv(KukaGymEnv(renders=True, isDiscrete=True, maxSteps=10000000))

# Reset the environment and launch the viewer
env.reset()
env.render()

# Step randomly until interrupted
steps = 0
while True:
    if steps == args.n_steps:
        break
    env.step(env.action_space.sample())
    steps += 1
