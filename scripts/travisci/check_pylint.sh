#!/usr/bin/env bash
if [[ "${TRAVIS_PULL_REQUEST}" != "false" && "${TRAVIS}" == "true" ]]; then
  git diff "${TRAVIS_COMMIT_RANGE}" --name-only \
    | xargs pylint --rcfile=setup.cfg
else
  git remote set-branches --add origin master
  git fetch
  git diff origin/master --name-only | xargs pylint --rcfile=setup.cfg
fi
