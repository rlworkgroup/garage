#!/usr/bin/env bash
set -e

nose2 -c setup.cfg -A cron_job
