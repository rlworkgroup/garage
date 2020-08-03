#!/usr/bin/env python3
"""Example of how to load, step, and visualize an environment.

This example requires that garage[dm_control] be installed.
"""
import argparse

from garage.envs.dm_control import DmControlEnv

parser = argparse.ArgumentParser()
parser.add_argument('--n_steps',
                    type=int,
                    default=1000,
                    help='Number of steps to run')
args = parser.parse_args()

# Construct the environment
env = DmControlEnv.from_suite('walker', 'run', args.n_steps)

# Reset the environment and launch the viewer
env.reset()
env.visualize()

# Step randomly until interrupted
while True:
    ts = env.step(env.action_space.sample())
    if ts.last:
        break
