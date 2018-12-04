#!/usr/bin/env bash
export TF_CPP_MIN_LOG_LEVEL=3      # shut TensorFlow up
export DISABLE_MUJOCO_RENDERING=1  # silence glfw
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${HOME}/.mujoco/mjpro150/bin"

status=0

if [[ "${TRAVIS_PULL_REQUEST}" != "false" && "${TRAVIS}" == "true" ]]; then
  pre-commit run --source "${TRAVIS_COMMIT_RANGE%...*}" \
                 --origin "${TRAVIS_COMMIT_RANGE#*...}"
  status="$((${status} | ${?}))"

  # Check commit messages
  while read commit; do
    commit_msg="$(mktemp)"
    git log --format=%B -n 1 "${commit}" > "${commit_msg}"
    pre-commit run --hook-stage commit-msg --commit-msg-file="${commit_msg}"
    pass=$?
    status="$((${status} | ${pass}))"

    # Print message if it fails
    if [[ "${pass}" -ne 0 ]]; then
      echo "Failing commit message:"
      cat "${commit_msg}"
    fi

  done < <(git rev-list "${TRAVIS_COMMIT_RANGE}")
else
  git remote set-branches --add origin master
  git fetch
  pre-commit run --source origin/master --origin ${TRAVIS_BRANCH}
  status="$((${status} | ${?}))"

  # Check commit messages
  while read commit; do
    commit_msg="$(mktemp)"
    git log --format=%B -n 1 "${commit}" > "${commit_msg}"
    pre-commit run --hook-stage commit-msg --commit-msg-file="${commit_msg}"
    pass=$?
    status="$((${status} | ${pass}))"

    # Print message if it fails
    if [[ "${pass}" -ne 0 ]]; then
      echo "Failing commit message:"
      cat "${commit_msg}"
    fi

  done < <(git rev-list origin/master..."${TRAVIS_BRANCH}")
fi

exit "${status}"
