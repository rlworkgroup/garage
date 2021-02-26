#!/bin/bash
cd /home/haydenshively
cp -r /home/avnishnarayan/.mujoco .
runuser -l haydenshively -c "git clone https://github.com/rlworkgroup/garage && cd garage/ && git checkout hayden-new-metaworld-results-st-v2 && mkdir data/"
runuser -l haydenshively -c "mkdir -p metaworld-runs-v2/local/experiment/"
runuser -l haydenshively -c "make run-headless -C ~/garage/"
runuser -l haydenshively -c "cd garage && python docker_metaworld_run_cpu.py 'metaworld_launchers/single_task_launchers/ppo_metaworld.py --env-name push-v2'"
runuser -l haydenshively -c "cd garage/metaworld_launchers && python upload_folders.py ppo_round3/push-v2 1200"
