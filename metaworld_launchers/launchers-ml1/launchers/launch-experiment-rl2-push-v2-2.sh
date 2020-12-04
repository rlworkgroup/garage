#!/bin/bash
cd /home/avnishnarayan
runuser -l avnishnarayan -c "git clone https://github.com/rlworkgroup/garage && cd garage/ && git checkout avnish-new-metaworld-results-mt1 && mkdir data/"
runuser -l avnishnarayan -c "mkdir -p metaworld-runs-v2/local/experiment/"
runuser -l avnishnarayan -c "make run-headless -C ~/garage/"
runuser -l avnishnarayan -c "cd garage && python docker_metaworld_run_cpu.py 'metaworld_launchers/ml1/rl2_ppo_metaworld_ml1.py --env-name push-v2'"
runuser -l avnishnarayan -c "cd garage/metaworld_launchers && python upload_folders.py ml1/round2/rl2/v2 1200"
