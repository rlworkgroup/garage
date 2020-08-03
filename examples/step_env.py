#!/usr/bin/env python3
"""Example of how to load, step, and visualize an environment."""
import argparse

import gym
from garage.envs import GarageEnv

parser = argparse.ArgumentParser()
parser.add_argument('--n_steps',
                    type=int,
                    default=1000,
                    help='Number of steps to run')
args = parser.parse_args()

# Construct the environment
env = GarageEnv(gym.make('MountainCar-v0'))

# Reset the environment and launch the viewer
env.reset()
env.visualize()

steps = 0
while True:
    if steps == args.n_steps:
        env.close()
        break
    env.step(env.action_space.sample())
    steps += 1
