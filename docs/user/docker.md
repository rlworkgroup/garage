# Run garage with Docker

Currently there are two types of garage images:

- headless: garage without environment visualization.
- nvidia: garage with environment visualization using an NVIDIA graphics
    card.

## Headless image

### Prerequisites

Be aware of the following prerequisites to build the image.

- Install [Docker CE](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)
  version 19.03 or higher. Tested on version 19.03.12.

Tested on Ubuntu 16.04, 18.04 & 20.04.

### Run a pre-compiled image

If you already have a source copy of garage, proceed to subsection [Build and
 run the headless image](#build-and-run-the-headless-image), otherwise, keep
reading.

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
  rlworkgroup/garage-headless python examples/tf/trpo_swimmer.py
```

To save the experiment data generated in the container, you need to specify a
path where the files will be saved inside your host computer with the argument
`-v` in the docker-run command. For example, if the path you want to use is
at `/home/tmp/data`, execute:

```
docker run \
  -it \
  --rm \
  -v /home/tmp/data:/home/garage-user/code/garage/data \
  rlworkgroup/garage-headless python examples/tf/trpo_cartpole.py
```

This binds a volume between your host path and the path in garage at
`/home/garage-user/code/garage/data`. Make sure the directory at the host
path exists, otherwise docker won't be able to write to it.

### Build and run the headless image

To build the headless image, first clone this repository, move to the root
folder of your local repository and then execute:

```
make build-headless
```

To build and run the container, execute;

```
make run-headless RUN_CMD="python examples/tf/trpo_cartpole.py"
```

Where RUN_CMD specifies the executable to run in the container.

The previous command adds a volume from the data folder inside your cloned
garage repository to the data folder in the garage container, so any experiment
results ran in the container will be saved in the data folder inside your
cloned repository. The data is saved in a folder named `data` inside the current
working directory. You can override that by using the variable `DATA_PATH` and
passing the *absolute path* to the output directory.

```
make run-headless RUN_CMD="..." DATA_PATH="/home/xyz/my_garage_exp"
```

By default, docker generates random names for containers. If you want to specify
a name for the container, do so with the variable `CONTAINER_NAME`.

```
make run-headless RUN_CMD="..." CONTAINER_NAME="my_container_123"
```

If you need to use MuJoCo, you need to place your key at `~/.mujoco/mjkey.txt`
or specify the corresponding path through the MJKEY_PATH variable:

```
make run-headless RUN_CMD="..." MJKEY_PATH="/home/user/mjkey.txt"
```

If you require to pass additional arguments to docker build and run commands,
you can use the variables BUILD_ARGS and RUN_ARGS, for example:

```
make build-headless BUILD_ARGS="--build-arg MY_VAR=123"
make run-headless RUN_ARGS="-e MY_VAR=123"
```

## NVIDIA image

The garage nvidia images come with CUDA 10.2.

### Prerequisites for nvidia image

Additional to the prerequisites for the headless image, make sure to have:

- Install the latest NVIDIA driver, tested
  on [nvidia-390](https://tecadmin.net/install-latest-nvidia-drivers-ubuntu/)
- [Install nvidia-docker2](https://github.com/NVIDIA/nvidia-docker#ubuntu-160418042004-debian-jessiestretchbuster)

Tested on Ubuntu 18.04 & 20.04.

### Run a pre-compiled garage-nvidia image

The same commands for the headless image mentioned above apply for the nvidia
image, except that the image name is defined by `rlworkgroup/garage-nvidia`.

For example, to execute a launcher file:

```
docker run -it --rm rlworkgroup/garage-nvidia python examples/tf/trpo_cartpole.py
```

### Build and run the nvidia image

The same rules for the headless image apply here, except that the target names
are the following:

```
make build-nvidia
make run-nvidia
```
