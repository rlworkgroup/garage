#!/usr/bin/env bash
./cc-test-reporter before-build
coverage run -m nose2 -c setup.cfg -E "not huge"
TEST_EXIT_CODE="$?"
coverage xml
bash <(curl -s https://codecov.io/bash)
coveralls
python-codacy-coverage -r coverage.xml
./cc-test-reporter after-build --exit-code $TEST_EXIT_CODE
