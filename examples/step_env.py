#!/usr/bin/env python3
"""Example of how to load, step, and visualize an environment."""
from garage.envs.dm_control import DmControlEnv

# Construct the environment
env = DmControlEnv.from_suite('walker', 'run')

# Reset the environment and launch the viewer
env.reset()
env.render()

# Step randomly until interrupted
try:
    print('Press Ctrl-C to stop...')
    while True:
        env.step(env.action_space.sample())
        env.render()
except KeyboardInterrupt:
    print('Exiting...')
    env.close()
