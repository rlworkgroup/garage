#!/usr/bin/env bash

coverage run -m nose2 -c setup.cfg -E 'not cron_job and not huge and not flaky'
coverage xml
bash <(curl -s https://codecov.io/bash)
