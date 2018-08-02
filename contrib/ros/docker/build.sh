#! /bin/bash

ROBOT=${1:-"sawyer"}

ROBOT_HOME="$HOME"/."$ROBOT"-deeprl-docker

mkdir -p $ROBOT_HOME

echo "Building"$ROBOT"-ros-docker"
if [ "$ROBOT" = "sawyer" ] ; then
  echo "Building sawyer-deeprl-docker..." ;
  docker build -f sawyer-Dockerfile \
    -t sawyer-deeprl-docker \
    --build-arg USER=$USER \
    --build-arg HOME=$HOME/.sawyer-deeprl-docker .;
    echo "GPU version of sawyer-deeprl-docker is not ready..." ;
else

fi
