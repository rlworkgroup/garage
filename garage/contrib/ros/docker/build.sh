#! /bin/bash

ROBOT=${1:-"sawyer"}

DEVICE=${2:-"cpu"}

if [ "$ROBOT" = "sawyer" ] ; then
  if [ "$DEVICE" = "cpu" ] ; then
    echo "Building sawyer-ros-docker with cpu..." ;
    docker build -f ./garage/contrib/ros/docker/sawyer/Dockerfile \
      -t sawyer-ros-docker:anaconda . ;
  else
    echo "sawyer-ros-docker with gpu is not ready yet..."
  fi
else
  echo "The robot "$ROBOT" is not supported by us!" ;
fi
