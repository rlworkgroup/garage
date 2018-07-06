#!/usr/bin/env bash
if [[ "${TRAVIS_PULL_REQUEST}" != "false" && "${TRAVIS}" == "true" ]]; then
  pre-commit run --source ${TRAVIS_COMMIT_RANGE%...*} --origin ${TRAVIS_COMMIT_RANGE#*...}
else
  git remote set-branches --add origin master
  git fetch
  pre-commit run --source origin/master --origin ${TRAVIS_BRANCH}
fi
