#!/usr/bin/env bash
set -e

coverage run -m nose2 -c setup.cfg -v --with-id -A large
coverage xml
bash <(curl -s https://codecov.io/bash)
