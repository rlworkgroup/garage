#!/bin/bash 
cd /home/avnishnarayan/
rm -rf garage/
rm -rf metaworld-runs-v2
runuser -l avnishnarayan -c "git clone https://github.com/rlworkgroup/garage && cd garage/ && git checkout run-maml-mt10 && mkdir data/"
runuser -l avnishnarayan -c "mkdir -p metaworld-runs-v2/local/experiment/"
runuser -l avnishnarayan -c "make run-headless -C ~/garage/"
runuser -l avnishnarayan -c "echo HERE1"
runuser -l avnishnarayan -c "ls"
runuser -l avnishnarayan -c "echo HERE2"
runuser -l avnishnarayan -c "cd garage && python docker_metaworld_run_cpu.py metaworld_launchers/ml1/maml_trpo_metaworld_ml1_pick_place.py && ls"
runuser -l avnishnarayan -c "cd garage/metaworld_launchers && python upload_folders.py ml1-pick-place 1200"

