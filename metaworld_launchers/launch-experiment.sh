#!/bin/bash 
cd /home/avnishnarayan/
rm -rf garage/
rm -rf metaworld-runs-v2
git clone https://github.com/rlworkgroup/garage
cd garage/
git checkout run-maml-mt10
mkdir data/
cd ../
mkdir -p metaworld-runs-v2/local/experiment/
runuser -l avnishnarayan -c "make run-headless -C ~/garage/"
runuser -l avnishnarayan -c "cd garage && python docker_metaworld_run_cpu.py metaworld_launchers/ml1/maml_trpo_metaworld_ml1_push.py"
# runuser -l avnishnarayan -c "cd garage/metaworld_launchers && python upload_folders.py -s ml1-push -t 1200"

