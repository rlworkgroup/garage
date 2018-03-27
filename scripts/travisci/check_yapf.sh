#!/usr/bin/env bash

if [[ "${TRAVIS_PULL_REQUEST}" != "false" ]]; then
  CHANGED=$(git diff --name-only ${TRAVIS_COMMIT_RANGE} | grep "\.py$")
else
  git remote set-branches --add origin integration
  git fetch
  CHANGED=$(git diff --name-only origin/integration | grep "\.py$")
fi

if [[ ! -z $CHANGED ]]; then
  echo "Only checking changes files: ${CHANGED}"
  yapf -dpr --style=pep8 $CHANGED
fi