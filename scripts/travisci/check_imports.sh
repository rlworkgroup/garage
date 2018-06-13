#!/usr/bin/env bash
export PYTHONPATH=.
export TF_CPP_MIN_LOG_LEVEL=3      # shut TensorFlow up
export DISABLE_MUJOCO_RENDERING=1  # silence glfw
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/.mujoco/mjpro150/bin"

python scripts/travisci/check_imports.py
