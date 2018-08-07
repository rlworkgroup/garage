FROM ubuntu:16.04
MAINTAINER Hejia Zhang <hjzh578@gmail.com>

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV USER=sawyer_docker HOME=/home/sawyer_docker
ENV ROS_WS=$HOME/ros_ws

RUN echo "The working directory is: $HOME"
RUN echo "The user is: $USER"

RUN mkdir -p $HOME

# apt dependencies
RUN \
  apt-get -y -q update && \
  # Prevents debconf from prompting for user input
  # See https://github.com/phusion/baseimage-docker/issues/58
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    # Dockerfile deps
    wget \
    bzip2 \
    unzip \
    git \
    # For building glfw
    cmake \
    xorg-dev \
    # Dummy X server
    xvfb \
    libosmesa6-dev \
    pulseaudio \
    # mujoco_py
    # See https://github.com/openai/mujoco-py/blob/master/Dockerfile
    # 16.04 repo is too old, install glfw from source instead
    # libglfw3 \
    libglew-dev \
    patchelf \
    # OpenAI gym
    # See https://github.com/openai/gym/blob/master/Dockerfile
    libpq-dev \
    ffmpeg \
    libjpeg-dev \
    swig \
    libsdl2-dev \
    # OpenAI baselines
    libopenmpi-dev \
    openmpi-bin && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Build GLFW because the Ubuntu 16.04 version is too old
# See https://github.com/glfw/glfw/issues/1004
RUN apt-get purge -y -v libglfw*
RUN git clone https://github.com/glfw/glfw.git && \
  cd glfw && \
  git checkout 0be4f3f75aebd9d24583ee86590a38e741db0904 && \
  mkdir glfw-build && \
  cd glfw-build && \
  cmake -DBUILD_SHARED_LIBS=ON -DGLFW_BUILD_EXAMPLES=OFF -DGLFW_BUILD_TESTS=OFF -DGLFW_BUILD_DOCS=OFF .. && \
  make -j"$(nproc)" && \
  make install && \
  cd ../../ && \
  rm -rf glfw

# MuJoCo
RUN mkdir /root/.mujoco && \
  wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip && \
  unzip mujoco.zip -d $HOME/.mujoco && \
  rm mujoco.zip
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin

# conda
RUN wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh && \
  bash miniconda.sh -b -p /opt/conda && \
  rm miniconda.sh && \
  ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ENV PATH=$PATH:/opt/conda/bin
RUN conda update -q -y conda

# conda environment
COPY environment.yml /root/code/garage/environment.yml
RUN conda env create -f /root/code/garage/environment.yml && \
    rm -rf /opt/conda/pkgs/*

# Extras
# prevent pip from complaining about available upgrades
RUN ["/bin/bash", "-c", "source activate garage && pip install --upgrade pip"]
# TensorFlow is not in environment.yml because of CPU vs GPU
RUN ["/bin/bash", "-c", "source activate garage && pip install tensorflow==1.8.0"]
# Fix Box2D install
# See https://github.com/openai/gym/issues/100
# See https://github.com/pybox2d/pybox2d/issues/82
RUN ["/bin/bash", "-c", "source activate garage && pip uninstall -y Box2D Box2D-kengz box2d-py && pip install Box2D"]

# Setup repo
WORKDIR /root/code/garage
# Pre-build pre-commit env
COPY .pre-commit-config.yaml /root/code/garage
RUN ["/bin/bash", "-c", "source activate garage && git init && pre-commit"]

# ROS
# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list
# setup keys
RUN apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
# install ros packages
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
  ros-kinetic-desktop-full
RUN rosdep init && rosdep update
# install bootstrap tools
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
  python-rosinstall \
  python-wstool \
  python-yaml

RUN ["/bin/bash", "-c", "source activate garage && pip install catkin_pkg rospkg defusedxml opencv-python empy"]

# Prepare Intera SDK installation
# Install dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
  git-core \
  python-argparse \
  python-wstool \
  python-vcstools \
  python-rosdep \
  ros-kinetic-control-msgs \
  ros-kinetic-joystick-drivers \
  ros-kinetic-xacro \
  ros-kinetic-tf2-ros \
  ros-kinetic-rviz \
  ros-kinetic-cv-bridge \
  ros-kinetic-actionlib \
  ros-kinetic-actionlib-msgs \
  ros-kinetic-dynamic-reconfigure \
  ros-kinetic-trajectory-msgs \
  ros-kinetic-rospy-message-converter

# Prepare sawyer gazebo simulator installation
# Install dependencies
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
  gazebo7 \
  ros-kinetic-qt-build \
  ros-kinetic-gazebo-ros-control \
  ros-kinetic-gazebo-ros-pkgs \
  ros-kinetic-ros-control \
  ros-kinetic-control-toolbox \
  ros-kinetic-realtime-tools \
  ros-kinetic-ros-controllers \
  ros-kinetic-xacro \
  python-wstool \
  ros-kinetic-tf-conversions \
  ros-kinetic-kdl-parser \
  ros-kinetic-sns-ik-lib

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y \
  libbullet-dev \
  python-empy

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y terminator

WORKDIR /root/code/garage

ENTRYPOINT ["garage/contrib/ros/docker/sawyer/docker-entrypoint.sh"]

# Add code stub last
COPY . /root/code/garage
ENV PYTHONPATH=$PYTHONPATH:/root/code/garage
