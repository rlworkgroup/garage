#! /bin/bash

ROBOT=${1:-"sawyer"}

if [ "$ROBOT" = "sawyer" ] ; then
  echo "Building sawyer-ros-docker..." ;
  nvidia-docker build -f ./garage/contrib/ros/docker/sawyer/Dockerfile \
    -t sawyer-ros-docker:anaconda . ;
else
  echo "The robot "$ROBOT" is not supported by us!" ;
fi
