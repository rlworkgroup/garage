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
	--volume=/media/:/media/:rw \
	--volume=$XSOCK:$XSOCK:rw \
	--volume=$XAUTH:$XAUTH:rw \
	--env="XAUTHORITY=${XAUTH}" \
	--env="USER_UID=${USER_UID}" \
	--env="USER_GID=${USER_GID}" \
	--env="DISPLAY=${DISPLAY}" \
	-p 6006:6006 \
	-p 8888:8888 \
	--cap-add SYS_ADMIN \
	--cap-add MKNOD \
	--device /dev/fuse \
	--security-opt apparmor:unconfined \
	--name "sawyer-ros-docker" \
  sawyer-ros-docker:anaconda;
else
  echo "The robot "$ROBOT" is not supported by us!";
fi

