import subprocess
import os
import click
import time

@click.command()
@click.option('--gpu', default=False, type=bool)
def launch_experiments(gpu):
    # entropies = [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 5e-5, 5e-5, 5e-5]
    entropies = [1e-4] * 10
    instances = [3, 4, 5] # [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]
    env_name = "push-v2"
    for i, entropy in enumerate(entropies):
        if not i % 4:
            instance_num = instances.pop(0)
        ####################EDIT THESE FIELDS##################
        username = f'avnishnarayan' # your google username
        algorithm = f'mtppo'
        zone = f'us-west1-a' # find the apprpropriate zone here https://cloud.google.com/compute/docs/regions-zones
        entropy_str = str(entropy).replace('.', '-')
        instance_name = f'mt1-{env_name}-v2-{algorithm}-{i}'
        bucket = f'mt1/{env_name}/{algorithm}/v2'
        branch = 'avnish-new-metaworld-results-mt1'
        experiment = f'metaworld_launchers/mt1/{algorithm}_metaworld_mt1.py --entropy {entropy} --env-name {env_name}'
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
