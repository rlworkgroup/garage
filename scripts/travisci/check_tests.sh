#!/usr/bin/env bash
set -e

coverage run -m nose2 -c setup.cfg -v --with-id -E 'not cron_job and not huge and not flaky'
coverage xml
bash <(curl -s https://codecov.io/bash)
