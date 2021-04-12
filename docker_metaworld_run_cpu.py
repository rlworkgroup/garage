import docker
from docker.types import DeviceRequest, Mount
import time
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='launch ml10 experiments')
parser.add_argument('run_cmd', metavar='r', type=str, help='command to be run')
args = parser.parse_args()
run_cmd = args.run_cmd

home = os.path.expanduser("~")

if not os.path.exists(f'{home}/metaworld-runs-v2'):
    os.makedirs(f'{home}/metaworld-runs-v2')

client = docker.from_env()

volume = Mount(f"{home}/code/garage/data", f"{home}/metaworld-runs-v2/", type='bind')
mjkey = open(f'{home}/.mujoco/mjkey.txt', 'r').read()
environment_vars = [f"MJKEY={mjkey}", "QT_X11_NO_MITSHM=1", "DISPLAY="]
device_requests = []

seeds = np.random.randint(10000,size=(1,))
for seed in seeds:
    client.containers.run('rlworkgroup/garage-dev',
                          f'python {run_cmd} --seed {seed}',
                          environment=environment_vars,
                          device_requests=device_requests,
                          mounts=[volume],
                          detach=True)
