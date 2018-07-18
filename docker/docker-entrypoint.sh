#!/usr/bin/env bash
# Dockerfile entrypoint
set -e

# Get MuJoCo key from the environment
echo "${MJKEY}" > /root/.mujoco/mjkey.txt

# Setup dummy X server display
display_num=99
export DISPLAY=:"${display_num}"
# xpra --xvfb="Xorg +extension GLX -config /root/code/garage/docker/dummy.xorg.conf -logfile /root/xorg.log" start "${DISPLAY}"
Xvfb -screen "${display_num}" 1024x768x24 &

# Wait for the file to come up
file="/tmp/.X11-unix/X${display_num}"
for i in $(seq 1 10); do
    if [ -e "$file" ]; then
  break
    fi

    echo "Waiting for $file to be created (try $i/10)"
    sleep "$i"
done
if ! [ -e "$file" ]; then
    echo "Timing out: $file was not created"
    exit 1
fi

# Activate conda environment
source activate garage

exec "$@"
