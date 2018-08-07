#!/bin/sh

ROBOT=${1:-"sawyer"}

USER_UID=$(id -u)
USER_GID=$(id -g)
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth -b nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -b -f $XAUTH nmerge -

if [ "$ROBOT" = "sawyer" ] ; then
  docker run \
	-it \
	--init \
	--volume=/home/:/home/:rw \
	--volume=$XSOCK:$XSOCK:rw \
	--volume=$XAUTH:$XAUTH:rw \
	--volume=/dev/bus/usb:/dev/bus/usb \
	--volume=/media:/media:rw \
	--env="XAUTHORITY=${XAUTH}" \
	--env="USER_UID=${USER_UID}" \
	--env="USER_GID=${USER_GID}" \
	--env="DISPLAY=${DISPLAY}" \
	--name "sawyer-ros-docker" \
	--privileged=true \
  sawyer-ros-docker:anaconda bash;
else
  echo "The robot "$ROBOT" is not supported by us!";
fi
