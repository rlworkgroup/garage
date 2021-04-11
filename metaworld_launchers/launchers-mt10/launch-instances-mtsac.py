import subprocess
import os
import click
import time

@click.command()
@click.option('--gpu', default=True, type=bool)
def launch_experiments(gpu):
    instances = [0, 1, 2]
    # instances = []
    # zones = ['europe-west3-b']
    zones = ['us-east1-c', 'us-east4-b', 'europe-west3-b', 'europe-west1-b']
    instance_threshs = [4, 4, 1, 1]
    instance_thresh = 1
    for i in range(10):
        if not i % instance_thresh:
            zone = zones.pop(0)
            instance_thresh = instance_threshs.pop(0)
        if not i % 4:
            instance_num = instances.pop(0)
        ####################EDIT THESE FIELDS##################
        username = f'avnishnarayan' # your google username
        algorithm = f'mtsac'
        zone = zone # find the apprpropriate zone here https://cloud.google.com/compute/docs/regions-zones
        instance_name = f'mt10-round5-v2-mtsac-{i}'
        bucket = f'mt10/round5/mtsac/v2'
        branch = 'avnish-new-metaworld-results-mt1'
        experiment = f'metaworld_launchers/mt10/mtsac_metaworld_mt10.py'
        ######################################################

        if not gpu:
            machine_type =  'n2-standard-8' # 'c2-standard-4' we have a quota of 24 of each of these cpus per zone. 
            # You can use n1 cpus which are slower, but we are capped to a total of 72 cpus per zone anyways
            docker_run_file = 'docker_metaworld_run_cpu.py' # 'docker_metaworld_run_gpu.py' for gpu experiment
            docker_build_command = 'make run-headless -C ~/garage/'
            source_machine_image = 'metaworld-v2-cpu-instance'
            launch_command = (f"gcloud beta compute instances create {instance_name} "
                f"--metadata-from-file startup-script=launchers/launch-experiment-{algorithm}-{i}.sh --zone {zone} "
                f"--source-machine-image {source_machine_image} --machine-type {machine_type}")
        else:
            machine_type =  'n1-highmem-8'
            docker_run_file = 'docker_metaworld_run_gpu.py'
            docker_build_command = ("make run-nvidia-headless -C ~/garage/ "
                '''PARENT_IMAGE='nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04' ''')
            source_machine_image = f'gpu-instance-{instance_num}'
            accelerator = '"type=nvidia-tesla-p100,count=1"'
            launch_command = (f"gcloud beta compute instances create {instance_name} "
                f"--metadata-from-file startup-script=launchers/launch-experiment-{algorithm}-{i}.sh --zone {zone} "
                f"--source-machine-image {source_machine_image} --machine-type {machine_type} "
                f'--accelerator={accelerator}')

        os.makedirs('launchers/', exist_ok=True)

        script = (
        "#!/bin/bash\n"
        f"cd /home/{username}\n"
        f'runuser -l {username} -c ""\n'
        f"rm -rf garage; rm -rf metaworld-runs-v2\n"
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
