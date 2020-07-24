#!/usr/bin/env python3
import os
import subprocess
from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_CLS_DICT

names = list(MEDIUM_MODE_CLS_DICT['train'].keys()) + list(MEDIUM_MODE_CLS_DICT['test'].keys()) 
for name in names:
    conda_args = [f'make run-headless RUN_ARGS="--cpus 4.5 --memory 10000m --memory-swap 0m" RUN_CMD="python examples/tf/ppo_metaworld.py --env_name {name}"']
    FNULL = open(os.devnull, 'w')
    p = subprocess.Popen(conda_args, shell=True, executable='/bin/bash', stdout=FNULL, stderr=subprocess.STDOUT)
    # p = subprocess.Popen(conda_args, shell=True, executable="/bin/bash",)
