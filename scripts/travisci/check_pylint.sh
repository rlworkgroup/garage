#!/usr/bin/env bash
if [[ "${TRAVIS_PULL_REQUEST}" != "false" && "${TRAVIS}" == "true" ]]; then
  files_changed=$(git diff "${TRAVIS_COMMIT_RANGE}" --name-only \
    | grep ".*.py")
  if [[ ! -z "${files_changed}" ]]; then
    pylint "${files_changed}" --rcfile=setup.cfg
  fi
else
  git remote set-branches --add origin master
  git fetch
  files_changed=$(git diff origin/master --name-only | grep ".*.py")
  if [[ ! -z "${files_changed}" ]]; then
    pylint "${files_changed}" --rcfile=setup.cfg
  fi
fi
