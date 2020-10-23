# Run garage with Docker

Currently there are two types of garage images available on Docker Hub:

- `garage`: garage without environment visualization.
- `garage-nvidia`: garage with environment visualization capability using an
 NVIDIA graphics card.

If you want to compile a new image using the the source, proceed to the document
[Building garage Docker image from source](docker_dev.md) instead.

# `garage` image

### Prerequisites

Be aware of the following prerequisites to run the image.

- Install [Docker CE](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)
  version 19.03 or higher. Tested on version 19.03.

Tested on Ubuntu 16.04, 18.04 and 20.04.

### Running the `garage` Docker image

The garage container comes bundled with the examples available in the garage
repo. To run an example launcher in the container, execute:

```bash
docker run -it --rm rlworkgroup/garage python examples/tf/trpo_cartpole.py
```

To get a list of all the examples, you can run:

```bash
docker run -it --rm rlworkgroup/garage ls -R examples
```

This will run the latest image available on Docker Hub, which coincides with
the latest stable release of garage. To use a specific release such as
v2020.06, use `rlworkgroup/garage:2020.06`.

The container runs with the user `garage-user` with the current working
directory as `/home/garage-user`.

To save the generated experiment data in a directory on your computer, you can
specify the path to that directory using the argument `-v` with the `docker run`
command. `-v` accepts an argument of the type `<absolute path of folder on
host machine>:<absolute path of folder inside container>`. Since the working
directory in the container is `/home/garage-user` by default, the results are
written to `/home/garage-user/data`.

For example, if the path of the directory on your computer where you want
the results to be stored is `/home/user/data`, make sure the directory exists.

Additionally, if you are on linux, make sure that this directory is writeable by
the container user `garage-user` by either making it accessible by running:

```bash
setfacl -m u:999:rwx /home/user/data
```

or making it writable to all other users by running:

```bash
chmod 777 /home/user/data
```

or giving ownership of the directory to `garage-user` through:

```bash
chown -R 999:docker /home/user/data
```

Then, if the path of the directory on your computer is `/home/user/data`,
execute:

```bash
docker run \
  -it \
  --rm \
  -v /home/user/data:/home/garage-user/data \
  rlworkgroup/garage \
  python examples/tf/trpo_cartpole.py
```

Similarly, if you want to run your own code inside the Docker container, you can
mount the directory containing your files as a volume, by using the `-v` option
with `docker run`. If your code is in a file called `launcher.py` in the
directory, say, `/home/user/my_garage_experiment_dir`, then you can mount it at
`/home/garage-user/my_experiment_dir` in the container as follows:

```bash
docker run \
  -it \
  --rm \
  -v /home/user/my_garage_experiment_dir:/home/garage-user/my_experiment_dir \
  -v /home/user/my_garage_experiment_dir/data:/home/garage-user/data \
  rlworkgroup/garage \
  python my_experiment/launcher123.py
```

To run environments using MuJoCo, pass the contents of the MuJoCo key in a
variable named MJKEY in the same docker-run command using `cat`. For example,
if your key is at `~/.mujoco/mjkey.txt`, execute:

```bash
docker run \
  -it \
  --rm \
  -e MJKEY="$(cat ~/.mujoco/mjkey.txt)" \
  # ... other arguments here
  rlworkgroup/garage \
  python examples/tf/trpo_swimmer.py
```

## `garage-nvidia` image

The garage NVIDIA images come with CUDA 10.2 and cuDNN 7.6 (required for
TensorFlow).

### Prerequisites for `garage-nvidia` image

Additional to the prerequisites for the headless image, make sure to have:

- [Install the latest NVIDIA driver](https://tecadmin.net/install-latest-nvidia-drivers-ubuntu/),
  tested on nvidia driver version 440.100. CUDA 10.2 requires a minimum of
  version 440.33. For older driver versions, see [Using a different driver
   version](#using-a-different-driver-version)
- [Install nvidia-container-runtime](https://github.com/NVIDIA/nvidia-container-runtime#installation)

Tested on Ubuntu 18.04 and 20.04.

### Running the `garage-nvidia` image

The same commands for the garage image mentioned above apply for the nvidia
image, except that the image name is `rlworkgroup/garage-nvidia`.

For example, to execute a launcher file:

```
docker run \
  -it \
  --rm \
  # ...
  rlworkgroup/garage-nvidia \
  python examples/tf/trpo_cartpole.py
```

### Enabling environment visualization

Allow the Docker container to access the X server on your machine by running:

```bash
xhost +local:docker
```

and while running the Docker container, add the following arguments to
`docker run`:

```bash
-v /tmp/.X11-unix:/tmp/.X11-unix
-e DISPLAY=$DISPLAY
-e QT_X11_NO_MITSHM=1
```

For example:

```bash
docker run \
  -it \
  --rm \
  -e MJKEY="$(cat ~/.mujoco/mjkey.txt)" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  rlworkgroup/garage \
  python examples/tf/trpo_swimmer.py
```


### Expose GPUs for use

By default, `garage-nvidia` uses all of your gpus. If you want to customize
which GPUs are used and/or want to set the GPU capabilities exposed, as
described in official Docker documentation
[here](https://docs.docker.com/config/containers/resource_constraints/#gpu),
you can pass the desired values to `--gpus` option as follows:

```
docker run \
  -it \
  --rm \
  --gpus "device=0,2" \
  # ...
  rlworkgroup/garage-nvidia \
  python examples/tf/trpo_cartpole.py
```

### Using a different driver version

The `garage-nvidia` Docker image uses `nvidia/cuda:10.2-cudnn7-runtime
-ubuntu18.04`
as the parent image which requires NVIDIA driver version 440.33+. If you need
to use garage with a different driver version, you might be able to build the
garage-nvidia image using a different parent image by following the guide at
[Building garage Docker image from source](docker_dev.md)

----

**This page was authored by Gitanshu Sardana ([@gitanshu](https://github.com
/gitanshu>)).**
