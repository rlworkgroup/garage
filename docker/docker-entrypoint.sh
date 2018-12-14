#!/usr/bin/env bash
# Dockerfile entrypoint
set -e

# Get MuJoCo key from the environment
echo "${MJKEY}" > /root/.mujoco/mjkey.txt

# Setup dummy X server display
display_num=0
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

# Activate conda environment
source activate garage

# Fixes Segmentation Fault
# See: https://github.com/openai/mujoco-py/pull/145#issuecomment-356938564
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/mesa/libGL.so.1

export TF_CPP_MIN_LOG_LEVEL=3      # shut TensorFlow up
export DISABLE_MUJOCO_RENDERING=1  # silence glfw
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/.mujoco/mjpro150/bin"

exec "$@"
