import subprocess
import os
import click
import time

@click.command()
@click.option('--gpu', default=False, type=bool)
def launch_experiments(gpu):
    entropies = [5e-3] * 10
    instances = [3, 4, 5]
    zones = ["us-west2-a", 'us-west2-a', "us-central1-a"]
    for i, entropy in enumerate(entropies):
        if not i % 8:
            zone = zones.pop(0)
        if not i % 4:
            instance_num = instances.pop(0)
        ####################EDIT THESE FIELDS##################
        username = f'avnishnarayan' # your google username
        algorithm = f'mttrpo'
        entropy_str = str(entropy).replace('.', '-')
        instance_name = f'v2-round6-mt10-mttrpo-tuned-entropy-{entropy_str}-{i}'
        bucket = f'mt10/round6/mttrpo/v2'
        branch = 'avnish-new-metaworld-results-mt1'
        experiment = f'metaworld_launchers/mt10/mttrpo_metaworld_mt10.py --entropy {entropy}'
        ######################################################

        if not gpu:
            machine_type =  'n1-standard-8' # 'c2-standard-4' we have a quota of 24 of each of these cpus per zone. 
            # You can use n1 cpus which are slower, but we are capped to a total of 72 cpus per zone anyways
            docker_run_file = 'docker_metaworld_run_cpu.py' # 'docker_metaworld_run_gpu.py' for gpu experiment
            docker_build_command = 'make run-headless -C ~/garage/'
            source_machine_image = f'cpu-instance-{instance_num}'
            launch_command = (f"gcloud beta compute instances create {instance_name} "
                f"--metadata-from-file startup-script=launchers/launch-experiment-{algorithm}-{i}.sh --zone {zone} "
                f"--source-machine-image {source_machine_image} --machine-type {machine_type}")
        else:
            machine_type =  'n1-standard-4'
            docker_run_file = 'docker_metaworld_run_gpu.py'
            docker_build_command = ("make run-nvidia-headless -C ~/garage/ "
                '''PARENT_IMAGE='nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04' ''')
            source_machine_image = 'metaworld-v2-gpu-instance'
            accelerator = '"type=nvidia-tesla-k80,count=1"'
            launch_command = (f"gcloud beta compute instances create {instance_name} "
                f"--metadata-from-file startup-script=launchers/launch-experiment-{algorithm}-{i}.sh --zone {zone} "
                f"--source-machine-image {source_machine_image} --machine-type {machine_type} "
                f'--accelerator={accelerator}')

        os.makedirs('launchers/', exist_ok=True)

        script = (
        "#!/bin/bash\n"
        f"cd /home/{username}\n"
        f'runuser -l {username} -c "git clone https://github.com/rlworkgroup/garage'
            f' && cd garage/ && git checkout {branch} && mkdir data/"\n'
        f'runuser -l {username} -c "mkdir -p metaworld-runs-v2/local/experiment/"\n'
        f'runuser -l {username} -c "{docker_build_command}"\n'
        f'''runuser -l {username} -c "cd garage && python {docker_run_file} '{experiment}'"\n'''
        f'runuser -l {username} -c "cd garage/metaworld_launchers && python upload_folders.py {bucket} 1200"\n')

        with open(f'launchers/launch-experiment-{algorithm}-{i}.sh', mode='w') as f:
            f.write(script)
        subprocess.Popen([launch_command], shell=True)
        print(launch_command)

launch_experiments()
