#! /bin/bash

ROBOT=${1:-"sawyer"}

ROBOT_HOME="$HOME"/."$ROBOT"-ros-docker

mkdir -p $ROBOT_HOME

if [ "$ROBOT" = "sawyer" ] ; then
  echo "Building sawyer-ros-docker..." ;
  docker build -f ./garage/contrib/ros/docker/sawyer-Dockerfile \
    -t sawyer-ros-docker:anaconda \
    --build-arg USER=$USER \
    --build-arg HOME=$HOME/.sawyer-ros-docker . ;
else
  echo "The robot "$ROBOT" is not supported by us!" ;
fi
