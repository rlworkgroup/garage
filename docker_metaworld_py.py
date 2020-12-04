import docker
from docker.types import DeviceRequest, Mount
import time
client = docker.from_env()

volume = Mount("/home/avnishn/code/garage/data", "/data/avnishn/metaworld-runs-2/", type='bind')

# names = [INSERT ENV NAMES HERE AS A LIST]
names = ['door-open-v2']

data = open('/home/avnishn/.mujoco/mjkey.txt', 'r').read()

environment_vars = [f"MJKEY={data}", "QT_X11_NO_MITSHM=1", "DISPLAY="]

device_requests = [DeviceRequest(count=-1, capabilities=[['gpu']])]
for gpu, name in enumerate(names):
    gpu = 0
    client.containers.run('rlworkgroup/garage-nvidia',
                          f'python examples/torch/sac_metaworld.py --env_name {name} --gpu {gpu}',
                          environment=environment_vars,
                          device_requests=device_requests,
                          mounts=[volume],
                          detach=False)
    time.sleep(0.1)
