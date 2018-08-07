#!/bin/sh

ROBOT=${1:-"sawyer"}

USER_UID=$(id -u)
USER_GID=$(id -g)

xhost +local:root

if [ -z ${NVIDIA_DRIVER+x} ]; then
	NVIDIA_DRIVER=$(nvidia-settings -q NvidiaDriverVersion | head -2 | tail -1 | sed 's/.*\([0-9][0-9][0-9]\)\..*/\1/') ;
fi
if [ -z ${NVIDIA_DRIVER+x} ]; then
	echo "Error: Could not determine NVIDIA driver version number. Please specify your driver version number manually in $0." 1>&2 ;
	exit ;
else
	echo "Linking to NVIDIA driver version $NVIDIA_DRIVER..." ;
fi

DOCKER_VISUAL_NVIDIA="-v /tmp/.X11-unix:/tmp/.X11-unix --device /dev/nvidia0 --device /dev/nvidiactl"


if [ "$ROBOT" = "sawyer" ] ; then
  nvidia-docker run \
	-it \
	--init \
	$DOCKER_VISUAL_NVIDIA \
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
  sawyer-ros-docker:anaconda bash;
else
  echo "The robot "$ROBOT" is not supported by us!";
fi

xhost -local:root

