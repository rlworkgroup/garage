#!/usr/bin/env bash
set -e

# Pre-commit checks
./scripts/travisci/check_precommit.sh

# Check normal-sized unit tests
coverage run -m nose2 -c setup.cfg -v --with-id -E 'not nightly and not huge and not flaky and not large'
coverage xml
bash <(curl -s https://codecov.io/bash)

# Ensure documentation still builds
pushd docs && make html && popd
