#!/usr/bin/env bash
if [[ "${TRAVIS_PULL_REQUEST}" != "false" && "${TRAVIS}" == "true" ]]; then
  files_changed=$(git diff "${TRAVIS_COMMIT_RANGE}" --name-only \
    | grep ".*\.py")
  if [[ ! -z "${files_changed}" ]]; then
    pre-commit run --files  ${files_changed}
  fi
else
  git remote set-branches --add origin master
  git fetch
  files_changed=$(git diff origin/master --name-only | grep ".*\.py")
  if [[ ! -z "${files_changed}" ]]; then
    pre-commit run --files ${files_changed}
  fi
fi
