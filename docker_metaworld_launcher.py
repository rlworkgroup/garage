import docker
from docker.types import DeviceRequest, Mount
import time
client = docker.from_env()

import os
from os.path import expanduser

home = expanduser("~")

if not os.path.exists(f'{home}/metaworld-runs-v2'):
    os.makedirs(f'{home}/metaworld-runs-v2')

volume = Mount(f"{home}/code/garage/data", f"{home}/metaworld-runs-v2/", type='bind')

# names = [INSERT ENV NAMES HERE AS A LIST]
names = ['door-open-v2']

data = open(f'{home}/.mujoco/mjkey.txt', 'r').read()

environment_vars = [f"MJKEY={data}", "QT_X11_NO_MITSHM=1", "DISPLAY="]

device_requests = [DeviceRequest(count=-1, capabilities=[['gpu']])]
for name in names:
    client.containers.run('rlworkgroup/garage-nvidia',
                          f'python metaworld_launchers/single_task_launchers/sac_metaworld.py --env_name {name}',
                          environment=environment_vars,
                          device_requests=device_requests,
                          mounts=[volume],
                          detach=True)
    time.sleep(0.1)
