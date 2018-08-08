#! /bin/bash

ROBOT=${1:-"sawyer"}

DEVICE=${2:-"cpu"}

if [ "$ROBOT" = "sawyer" ] ; then
  if [ "$DEVICE" = "cpu" ] ; then
    echo "Building sawyer-ros-docker with cpu..." ;
    docker build -f ./garage/contrib/ros/docker/sawyer/Dockerfile \
      -t sawyer-ros-docker:cpu \
      --build-arg DOCKER_FROM="ubuntu:16.04" . ;
  else
    echo "Building sawyer-ros-docker with gpu..." ;
    docker build -f ./garage/contrib/ros/docker/sawyer/Dockerfile \
      -t sawyer-ros-docker:gpu \
      --build-arg DOCKER_FROM="tensorflow/tensorflow:1.8.0-gpu-py3" . ;
  fi
else
  echo "The robot "$ROBOT" is not supported by us!" ;
fi
