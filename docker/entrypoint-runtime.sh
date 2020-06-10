#!/usr/bin/env bash
# Dockerfile entrypoint
set -e

# Get MuJoCo key from the environment
if [ -z "${MJKEY}" ]; then
  :
else
  echo "${MJKEY}" > ${HOME}/.mujoco/mjkey.txt
fi

exec "$@"
