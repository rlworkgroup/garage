#!/bin/bash
cd /home/avnishnarayan
runuser -l avnishnarayan -c "git clone https://github.com/rlworkgroup/garage && cd garage/ && git checkout avnish-new-metaworld-results-mt1 && mkdir data/"
runuser -l avnishnarayan -c "mkdir -p metaworld-runs-v2/local/experiment/"
runuser -l avnishnarayan -c "make run-headless -C ~/garage/"
runuser -l avnishnarayan -c "cd garage && python docker_metaworld_run_cpu.py 'metaworld_launchers/mt50/mttrpo_metaworld_mt50.py --entropy 0.005'"
runuser -l avnishnarayan -c "cd garage/metaworld_launchers && python upload_folders.py mt50/round1/mttrpo/v2 1200"
