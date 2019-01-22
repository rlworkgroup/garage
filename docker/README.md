# Using Docker to run garage

Currently there are two types of garage images:
  - headless: garage without environment visualization.
  - nvidia: garage with environment visualization using an NVIDIA graphics
    card.

## Headless image

If you already have a copy of garage, proceed to subsection ["Build and run the
image"], otherwise, keep reading.

To run an example launcher in the container, execute:
```
docker run -it --rm rlworkgroup/garage-headless python examples/tf/trpo_cartpole.py
```

To run environments using MuJoCo, pass the contents of the MuJoCo key in a
variable named MJKEY in the same docker-run command using `cat`. For example,
if your key is at `~/.mujoco/mjkey.txt`, execute:
```
docker run \
  -it \
  --rm \
  -e MJKEY="$(cat ~/.mujoco/mjkey.txt)" \
  rlworkgroup/garage-headless python examples/theano/trpo_swimmer.py
```

To save the experiment data generated in the container, you need to specify a
path where the files will be saved inside your host computer with the argument
`-v` in the docker-run command. For example, if the path you want to use is
at `/home/tmp/data`, execute:
```
docker run \
  -it \
  --rm \
  -v /home/tmp/data:/root/code/garage/data \
  rlworkgroup/garage-headless python examples/tf/trpo_cartpole.py
```
This binds a volume between your host path and the path in garage at
`/root/code/garage/data`.

### Build and run the image

To build the headless image, first clone this repository and move to the root
folder of your local repository. If you have your MuJoCo key at
`~/.mujoco/mjkey.txt`, simply execute:
```
make build-headless
```

Alternatively, you can specify the path to your MuJoCo key with the variable
`MJKEY_PATH`:
```
make build-headless MJKEY_PATH="..."
```

To build and run the container, execute;
```
make run-headless RUN_CMD="python examples/tf/trpo_cartpole.py"
```
Where RUN_CMD specifies the executable to run in the container

The previous command adds a volume from the data folder inside your cloned
garage repository to the data folder in the garage container, so any experiment
results ran in the container will be saved in the data folder inside your
cloned repository. The data is saved in a folder with the name of the container
that generated the data, which by default is the name of the image type the
container is based on with the date and time the container was launched.

If you want to specify another name for the container, do so with the variable
`CONTAINER_NAME`:
```
make run-headless RUN_CMD="..." CONTAINER_NAME="my_container"
```

#### Prerequisites

Be aware of the following prerequisites to build the image.

- Install [Docker CE](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)
- Install [Docker Compose](https://docs.docker.com/compose/install/#install-compose)

Tested on Ubuntu 16.04.

## nvidia image

The same commands for the headless image mentioned above apply for the nvidia
image, except that the image name is defined by `rlworkgroup/garage-nvidia`.
For example, to execute a launcher file:
```
docker run -it --rm rlworkgroup/garage-nvidia python examples/tf/trpo_cartpole.py
```

### Build and run the image

The same rules for the headless image apply here, except that the target names
are the following:
```
make build-nvidia
make run-nvidia
```

#### Prerequisites

Additional to the prerequisites for the headless image, make sure to have:
- Install the latest NVIDIA driver, tested
  on [nvidia-390](https://tecadmin.net/install-latest-nvidia-drivers-ubuntu/)
- [Install nvidia-docker2](https://github.com/NVIDIA/nvidia-docker#ubuntu-140416041804-debian-jessiestretch)

Tested on Ubuntu 16.04.
