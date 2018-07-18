#!/usr/bin/env bash
# Dockerfile entrypoint
set -e

# Get MuJoCo key from the environment
echo "${MJKEY}" | /root/.mujoco/mjkey.txt

# Setup dummy X server display
export DISPLAY=:99
xpra --xvfb=\"Xorg +extension GLX -config /root/dummy.xorg.conf -logfile /root/xorg.log\"  start "${DISPLAY}"

exec "@"

# Stop the dummy X server display
xpra stop "${DISPLAY}"
