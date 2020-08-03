ARG PARENT_IMAGE=ubuntu:18.04

# Garage base target ###########################################################
FROM $PARENT_IMAGE AS garage-18.04

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# Add PPA for old Python 3.5
RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Dockerfile deps
    software-properties-common && \
  add-apt-repository ppa:deadsnakes/ppa

# apt dependencies
RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Dockerfile deps
    wget \
    unzip \
    git \
    curl \
    # For building glfw
    cmake \
    xorg-dev \
    # mujoco_py
    # See https://github.com/openai/mujoco-py/blob/master/Dockerfile
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    # OpenAI baselines
    libopenmpi-dev \
    # virtualenv
    python3.5 \
    python3.5-dev \
    python3.5-tk \
    python3.5-venv && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Install pip on system since python3.5-dev package doesn't install it by default.
# This is different from a similar command around line 104 since that upgrades
# the pip module inside virtual env
RUN python3.5 -m ensurepip && \
  python3.5 -m pip install --upgrade pip setuptools

# Build GLFW because the Ubuntu 18.04 version is too old
# See https://github.com/glfw/glfw/issues/1004
RUN wget https://github.com/glfw/glfw/releases/download/3.3/glfw-3.3.zip && \
  unzip glfw-3.3.zip && \
  rm glfw-3.3.zip && \
  cd glfw-3.3 && \
  mkdir glfw-build && \
  cd glfw-build && \
  cmake -DBUILD_SHARED_LIBS=ON -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF .. && \
  make -j"$(nproc)" && \
  make install && \
  cd ../../ && \
  rm -rf glfw-3.3

ARG user=garage-user
ARG uid=999
RUN groupadd -g $uid $user && \
    useradd -m -r -u $uid -g $user $user && \
    chown -R $user:$user /home/$user
USER $user
ENV HOME /home/$user
ENV PATH "$PATH:$HOME/.local/bin"
ENV PATH_NO_VENV $PATH
WORKDIR $HOME

# MuJoCo 2.0 (for dm_control)
RUN mkdir -p $HOME/.mujoco && \
  wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip --no-check-certificate && \
  unzip mujoco.zip -d $HOME/.mujoco && \
  rm mujoco.zip && \
  ln -s $HOME/.mujoco/mujoco200_linux $HOME/.mujoco/mujoco200
# This is a hack required to make mujoco-py compile in gpu mode
USER root
RUN mkdir -p /usr/lib/nvidia-000
USER $user
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin:/usr/lib/nvidia-000

# Fixes Segmentation Fault
# See: https://github.com/openai/mujoco-py/pull/145#issuecomment-356938564
ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libGLEW.so

# Set MuJoCo rendering mode (for dm_control)
ENV MUJOCO_GL "glfw"


# Create virtualenv
ENV VIRTUAL_ENV $HOME/venv
RUN python3.5 -m venv $VIRTUAL_ENV
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

# Prevent pip from complaining about available upgrades inside virtualenv
RUN pip install --upgrade pip setuptools wheel && \
  rm -r $HOME/.cache/pip

# We need a MuJoCo key to install mujoco_py
# In this step only the presence of the file mjkey.txt is required, so we only
# create an empty file
RUN touch $HOME/.mujoco/mjkey.txt

COPY --chown=$user:$user docker/entrypoint-runtime.sh $HOME/code/garage/docker/entrypoint-runtime.sh
ENTRYPOINT ["code/garage/docker/entrypoint-runtime.sh"]


# Headless machine with xvfb, i.e. so rendering works ##########################
FROM garage-18.04 AS garage-headless-18.04
USER root
RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Dummy X server
    xvfb \
    pulseaudio && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  mkdir -p -m 1777 /tmp/.X11-unix

ARG user=garage-user
ARG uid=999
USER $user

# Ready, set, go.
COPY --chown=$user:$user docker/entrypoint-headless.sh $HOME/code/garage/docker/entrypoint-headless.sh
ENTRYPOINT ["code/garage/docker/entrypoint-headless.sh"]


# Special setup for garage developers, which installs garage from source #######
# Assumes $PWD is the garage repository.
FROM garage-headless-18.04 AS garage-dev-18.04

ARG user=garage-user
ARG uid=999
# Copy over just setup.py first, so the Docker cache doesn't expire until
# dependencies change
#
# Files needed to run setup.py
# - README.md
# - VERSION
# - scripts/garage
# - src/garage/__init__.py
# - setup.py
COPY --chown=$user:$user README.md $HOME/code/garage/README.md
COPY --chown=$user:$user VERSION $HOME/code/garage/VERSION
COPY --chown=$user:$user scripts/garage $HOME/code/garage/scripts/garage
COPY --chown=$user:$user src/garage/__init__.py $HOME/code/garage/src/garage/__init__.py
COPY --chown=$user:$user setup.py $HOME/code/garage/setup.py
WORKDIR $HOME/code/garage

# Pre-build pre-commit env
COPY --chown=$user:$user .pre-commit-config.yaml $HOME/code/garage
RUN git init && \
  pip install pre-commit && \
  pre-commit install && \
  pre-commit install-hooks && \
  rm -r $HOME/.cache/pip

# Install deps (but not the code)
RUN pip install --upgrade pip setuptools wheel && \
  pip install .[all,dev] && \
  rm $HOME/.mujoco/mjkey.txt && \
  rm -r $HOME/.cache/pip

# Add code stub last (ensures code changes have the shortest builds)
COPY --chown=$user:$user . $HOME/code/garage

# Build and install the sdist
RUN python3.5 setup.py sdist && \
    cp dist/*.tar.gz dist/garage.tar.gz && \
    pip install dist/garage.tar.gz[all,dev]

RUN cd benchmarks && python3.5 setup.py sdist && \
    cp dist/*.tar.gz dist/benchmarks.tar.gz && \
    pip install dist/benchmarks.tar.gz

ENTRYPOINT ["docker/entrypoint-headless.sh"]

# Nvidia target ################################################################
FROM garage-dev-18.04 AS garage-nvidia-18.04
ENTRYPOINT ["docker/entrypoint-runtime.sh"]

# Test target ##################################################################
FROM garage-dev-18.04 AS garage-test-18.04
CMD nice -n 11 pytest -v -n auto -m 'not huge and not flaky' --durations=20
