#!/usr/bin/env bash
# Dockerfile entrypoint
set -e

# Get MuJoCo key from the environment
echo "${MJKEY}" > /root/.mujoco/mjkey.txt

# Fixes Segmentation Fault
# See: https://github.com/openai/mujoco-py/pull/145#issuecomment-356938564
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# Set MuJoCo rendering mode (for dm_control)
export MUJOCO_GL="glfw"

export TF_CPP_MIN_LOG_LEVEL=3      # shut TensorFlow up
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/.mujoco/mujoco200/bin"

exec "$@"
