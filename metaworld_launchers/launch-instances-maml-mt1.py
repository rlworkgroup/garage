import subprocess
import os

algorithm ='maml-trpo'
sub_experiment_type = 'mamled-baseline'
experiment_type = f'mt1-reach-tuning-{sub_experiment_type}'
num_experiments = 3
zones =  ['us-east4-a',]
machine_types = ['n2-standard-8', 'c2-standard-8']
source_machine_image = 'metaworld-v2-cpu-instance'
bucket = f'mt1-reach-maml-tuning/{sub_experiment_type}'
branch='avnish-maml-mlp-baseline'

lr = [1e-3, 5e-3, 1e-2, 1e-1, 2e-1, 5e-1]
for i, zone in enumerate(zones):
    for j, machine_type in enumerate(machine_types):
        for k in range(3):
            script_number = (i  * len(machine_types) * 3) + j * 3 + k
            script = (
f'''#!/bin/bash
cd /home/avnishnarayan/
rm -rf garage/
rm -rf metaworld-runs-v2
runuser -l avnishnarayan -c "git clone https://github.com/rlworkgroup/garage && cd garage/ && git checkout {branch} && mkdir data/"
runuser -l avnishnarayan -c "mkdir -p metaworld-runs-v2/local/experiment/"
runuser -l avnishnarayan -c "make run-headless -C ~/garage/"
runuser -l avnishnarayan -c "cd garage && python docker_metaworld_run_cpu.py 'metaworld_launchers/mt1/maml_trpo_metaworld_mt1.py --env-name reach-v2 --inner_lr {lr[script_number]}'"
runuser -l avnishnarayan -c "cd garage/metaworld_launchers && python upload_folders.py {bucket} 1200"''')
            with open(f'launchers/launch-experiment-maml-mt1-{script_number}.sh', mode='w') as f:
                f.write(script)
            instance_h_params = str(lr[script_number]).replace('.', '-')
            instance_name = f'{algorithm}-{experiment_type}-inner-lr-{instance_h_params}'
            subprocess.Popen([f"gcloud beta compute instances create {instance_name} --metadata-from-file startup-script=launchers/launch-experiment-maml-mt1-{script_number}.sh --zone {zone} --source-machine-image {source_machine_image} --machine-type {machine_type}"], shell=True)
