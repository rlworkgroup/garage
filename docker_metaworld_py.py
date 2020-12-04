import docker
from docker.types import DeviceRequest, Mount
import time
client = docker.from_env()

volume = Mount("/home/avnishn/code/garage/data", "/data/avnishn/metaworld-runs-2/", type='bind')

# names = ['drawer-close-v2', 'dial-turn-v2', 'reach-v2', 'window-open-v2', 'window-close-v2', 'lever-pull-v2', 'drawer-open-v2', 'door-close-v2', 'peg-insert-side-v2']
names = ['peg-insert-side-v2']

data = open('/home/avnishn/.mujoco/mjkey.txt', 'r').read()

environment_vars = [f"MJKEY={data}", "QT_X11_NO_MITSHM=1", "DISPLAY="]

device_requests = [DeviceRequest(count=-1, capabilities=[['gpu']])]
# for gpu, name in enumerate(names):
#     client.containers.run('rlworkgroup/garage-nvidia',
#                           f'python examples/torch/sac_metaworld.py --env_name {name} --gpu {gpu}',
#                           environment=environment_vars,
#                           device_requests=device_requests,
#                           mounts=[volume],
#                           detach=False)
#     time.sleep(0.1)
name = 'peg-insert-side-v2'
gpu = 0
client.containers.run('rlworkgroup/garage-nvidia',
                          f'python examples/torch/sac_metaworld.py --env_name {name} --gpu {gpu}',
                          environment=environment_vars,
                          device_requests=device_requests,
                          mounts=[volume],
                          detach=False,
                          stdout=True,
                          stderror=True)