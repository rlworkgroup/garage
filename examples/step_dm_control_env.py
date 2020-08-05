#!/usr/bin/env python3
"""Example of how to load, step, and visualize an environment.

This example requires that garage[dm_control] be installed.
"""
import argparse

from garage.envs.dm_control import DMControlEnv

parser = argparse.ArgumentParser()
parser.add_argument('--n_steps',
                    type=int,
                    default=1000,
                    help='Number of steps to run')
args = parser.parse_args()

# Construct the environment
env = DMControlEnv.from_suite('walker', 'run')

# Reset the environment and launch the viewer
env.reset()
env.visualize()

# Step randomly until interrupted
step_count = 0
es = env.step(env.action_space.sample())
while not es.last and step_count < args.n_steps:
    es = env.step(env.action_space.sample())
    step_count += 1

env.close()
