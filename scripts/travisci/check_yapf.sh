#!/usr/bin/env bash
CHANGED=$(git diff --name-only $TRAVIS_COMMIT_RANGE | grep "\.py$")
if [[ ! -z $CHANGED ]]; then
  yapf -dpr --style=pep8 $CHANGED
fi