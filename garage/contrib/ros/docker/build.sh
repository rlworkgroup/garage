#! /bin/bash

ROBOT=${1:-"sawyer"}

DEVICE=${2:-"cpu"}

if [ "$ROBOT" = "sawyer" ] ; then
  if [ "$DEVICE" = "cpu" ] ; then
    echo "Building sawyer-ros-docker with cpu..." ;
    docker build -f ./garage/contrib/ros/docker/sawyer/Dockerfile.cpu \
      -t sawyer-ros-docker:cpu . ;
  else
    echo "Building sawyer-ros-docker with gpu..." ;
    docker build -f ./garage/contrib/ros/docker/sawyer/Dockerfile.gpu \
      -t sawyer-ros-docker:gpu . ;
  fi
else
  echo "The robot "$ROBOT" is not supported by us!" ;
fi
