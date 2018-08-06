#!/bin/bash

export USER=sawyer_docker

# Clean up bashrc
echo "export USER=sawyer_docker" > "/home/$USER/.bashrc"

echo "source /opt/ros/kinetic/setup.bash" >> "/home/$USER/.bashrc"
source /opt/ros/kinetic/setup.bash

echo "export USER=$USER" >> "/home/$USER/.bashrc"
echo "export HOME=/home/$USER" >> "/home/$USER/.bashrc"

export HOME=/home/$USER

if [ ! -d "/home/$USER/ros_ws/src" ]; then
  mkdir -p "/home/$USER/ros_ws/src"
fi

# Prepare install
