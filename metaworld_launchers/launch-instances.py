import subprocess
import os
import click

@click.command()
@click.option('--gpu', default=False, type=bool)
def launch_experiments(gpu):
    ####################EDIT THESE FIELDS##################
    username = f'' # your google username
    algorithm = f'sac'
    zone = f'' # find the apprpropriate zone here https://cloud.google.com/compute/docs/regions-zones
    instance_name = f'' # name that you want to give the instance. Can only be 63 chars long, no special chars except '-'
    bucket = f'sac/'
    branch = ''
    experiment = 'metaworld_launchers/single_task_launchers/sac_metaworld.py --env-name reach-v2'
    ######################################################

    if not gpu:
        machine_type =  'n2-standard-4' # 'c2-standard-4' we have a quota of 24 of each of these cpus per zone. 
        # You can use n1 cpus which are slower, but we are capped to a total of 72 cpus per zone anyways
        docker_run_file = 'docker_metaworld_run_cpu.py' # 'docker_metaworld_run_gpu.py' for gpu experiment
        docker_build_command = 'make run-headless -C ~/garage/' 
        source_machine_image = 'metaworld-v2-cpu-instance'
        launch_command = (f"gcloud beta compute instances create {instance_name} "
            f"--metadata-from-file startup-script=launch-experiment-.sh --zone {zone} "
            f"--source-machine-image {source_machine_image} --machine-type {machine_type}")
    else:
        machine_type =  'n1-standard-4' 
        docker_run_file = 'docker_metaworld_run_gpu.py'
        docker_build_command = ("make run-nvidia-headless -C ~/garage/ " 
            '''PARENT_IMAGE='nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04'"''')
        source_machine_image = 'metaworld-v2-gpu-instance'
        accelerator = '[count=1],[type=nvidia-tesla-k80]'
        launch_command = (f"gcloud beta compute instances create {instance_name} "
            f"--metadata-from-file startup-script=launch-experiment-.sh --zone {zone} "
            f"--source-machine-image {source_machine_image} --machine-type {machine_type}"
            f"--accelerator {accelerator}")

    os.makedirs('launchers/', exist_ok=True)

    script = (
    "#!/bin/bash\n"
    f"cd /home/{username}\n/"
    f'runuser -l {username} -c "git clone https://github.com/rlworkgroup/garage'
        f' && cd garage/ && git checkout {branch} && mkdir data/"\n'
    f'runuser -l {username} -c "mkdir -p metaworld-runs-v2/local/experiment/"\n'
    f'runuser -l {username} -c "{docker_build_command}"\n'
    f'''runuser -l {username} -c "cd garage && python {docker_run_file} '{experiment}'"\n'''
    f'runuser -l {username} -c "cd garage/metaworld_launchers && python upload_folders.py {bucket} 1200"\n')

    with open(f'launchers/launch-experiment.sh', mode='w') as f:
        f.write(script)

    subprocess.run([launch_command], shell=True)

launch_experiments()