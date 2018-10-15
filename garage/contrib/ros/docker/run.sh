#!/bin/sh

ROBOT=${1:-"sawyer"}

USER_UID=$(id -u)
USER_GID=$(id -g)

xhost +local:root

DOCKER_VISUAL="-v /tmp/.X11-unix:/tmp/.X11-unix"


if [ "$ROBOT" = "sawyer" ] ; then
  docker run \
	-it \
	--rm \
	--init \
	$DOCKER_VISUAL \
	--env="USER_UID=${USER_UID}" \
	--env="USER_GID=${USER_GID}" \
	--env="USER=${USER}" \
	--env="DISPLAY" \
	--env="QT_X11_NO_MITSHM=1" \
	--volume=/home/:/home/:rw \
	--volume=/dev/bus/usb:/dev/bus/usb:ro \
	--volume=/media:/media:rw \
	--cap-add SYS_ADMIN \
	--cap-add MKNOD \
	--device /dev/fuse \
	--name "sawyer-ros-docker" \
	--security-opt apparmor:unconfined \
  sawyer-ros-docker:cpu bash;
else
  echo "The robot "$ROBOT" is not supported by us!";
fi

xhost -local:root
