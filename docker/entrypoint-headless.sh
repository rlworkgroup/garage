#!/usr/bin/env bash
# Dockerfile entrypoint
set -e

# Get MuJoCo key from the environment
echo "${MJKEY}" > /root/.mujoco/mjkey.txt

# Setup dummy X server display
# Socket for display :0 may already be in use if the container is connected
# to the network of the host, and other low-numbered socket could also be in
# use, that's why we use 100.
display_num=100
export DISPLAY=:"${display_num}"
Xvfb "${DISPLAY}" -screen 0 1024x768x24 &
pulseaudio -D --exit-idle-time=-1

# Wait for X to come up
file="/tmp/.X11-unix/X${display_num}"
for i in $(seq 1 10); do
  if [ -e "$file" ]; then
    break
  fi
  echo "Waiting for X to start (i.e. $file to be created) (attempt $i/10)"
  sleep "$i"
done
if ! [ -e "$file" ]; then
  echo "Timed out waiting for X to start: $file was not created"
  exit 1
fi

# Fixes Segmentation Fault
# See: https://github.com/openai/mujoco-py/pull/145#issuecomment-356938564
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# Set MuJoCo rendering mode (for dm_control)
export MUJOCO_GL="glfw"

export TF_CPP_MIN_LOG_LEVEL=3      # shut TensorFlow up
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco200/bin"

exec "$@"
