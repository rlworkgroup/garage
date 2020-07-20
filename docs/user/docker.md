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

This will run the latest image available. To use a stable release such as
v2020.06, use `rlworkgroup/garage-headless:2020.06`.

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
`/home/garage-user/code/garage/data`.

``` note:: Make sure the directory at the host path exists and is writable by
 the current user, otherwise docker will create it with user as root, but the
 garage container won't be able to write to it.
```

### Build and run the headless image

To build and run the headless image, first clone the garage repository,
move to the root folder of your local repository and then execute;

```
make run-headless RUN_CMD="python examples/tf/trpo_cartpole.py"
```

Where RUN_CMD specifies the executable to run in the container.

The previous command adds a volume from the `data` folder inside your cloned
garage repository to the `data` folder in the garage container, so any
experiment results ran in the container will be saved in the `data` folder
inside your cloned repository.

By default, docker generates random names for containers. If you want to specify
a name for the container, you can do so with the variable `CONTAINER_NAME`. As a
side effect, this will output the results in `data/$CONTAINER_NAME` directory
instead of the `data` directory.

```
make run-headless RUN_CMD="..." CONTAINER_NAME="my_container_123"
```

This will output results in `data/my_container_123` directory.

If you need to use MuJoCo, you need to place your key at `~/.mujoco/mjkey.txt`
or specify the corresponding path through the MJKEY_PATH variable:

```
make run-headless RUN_CMD="..." MJKEY_PATH="/home/user/mjkey.txt"
```

If you require to pass additional arguments to docker build and run commands,
you can use the variables BUILD_ARGS and RUN_ARGS, for example:

```
make run-headless BUILD_ARGS="--build-arg MY_VAR=123" RUN_ARGS="-e MY_VAR=123"
```

## NVIDIA image

The garage NVIDIA images come with CUDA 10.2.

### Prerequisites for NVIDIA image

Additional to the prerequisites for the headless image, make sure to have:

- [Install the latest NVIDIA driver](https://tecadmin.net/install-latest-nvidia-drivers-ubuntu/),
  tested on nvidia driver version 440.100. CUDA 10.2 requires a minimum of
  version 440.33. For older driver versions, see [Using a different driver
   version](#using-a-different-driver-version)
- [Install nvidia-container-runtime](https://github.com/NVIDIA/nvidia-container-runtime#installation)

Tested on Ubuntu 18.04 & 20.04.

### Run a pre-compiled garage-nvidia image

The same commands for the headless image mentioned above apply for the nvidia
image, except that the image name is defined by `rlworkgroup/garage-nvidia`.

For example, to execute a launcher file:

```
docker run -it --rm rlworkgroup/garage-nvidia python examples/tf/trpo_cartpole.py
```

### Build and run the NVIDIA image

The same rules for the headless image apply here, except that the target name
is:

```
make run-nvidia
```

This make command builds the NVIDIA image and runs it in a non-headless mode.
It will not work on headless machines. You can run the NVIDIA in a headless
state using the following target:

```
make run-nvidia-headless
```

### Expose GPUs for use

By default, garage-nvidia uses all of your gpus. If you want to customize which
GPUs are used and/or want to set the GPU capabilities exposed, as described in
official docker documentation
[here](https://docs.docker.com/config/containers/resource_constraints/#gpu),
you can pass the desired values to --gpus option using the variable GPUS. For
example:

```
make run-nvidia GPUS="device=0,2"
```

### Using a different driver version

The garage-nvidia docker image uses `nvidia/cuda:10.2-runtime-ubuntu18.04` as
the parent image which requires NVIDIA driver version 440.33+. If you need
to use garage with a different driver version, you might be able to build the
garage-nvidia image from scratch using a different parent image using the
variable `PARENT_IMAGE`.

```
make run-nvidia PARENT_IMAGE="nvidia/cuda:10.1-runtime-ubuntu18.04"
```

You can find the required parent images at [NVIDIA CUDA's DockerHub](https://hub.docker.com/r/nvidia/cuda/tags)

----

This page was authored by Angel Ivan Gonzalez ([@gonzaiva](https://github.com/gonzaiva)), with contributions from Gitanshu Sardana ([@gitanshu](https://github.com/gitanshu>)).
