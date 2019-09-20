#!/usr/bin/env bash
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
  merge_base=$(git merge-base origin/master ${TRAVIS_BRANCH})
  pre-commit run --source ${merge_base} --origin ${TRAVIS_BRANCH}
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

  done < <(git rev-list ^origin/master "${TRAVIS_BRANCH}")
fi

exit "${status}"
