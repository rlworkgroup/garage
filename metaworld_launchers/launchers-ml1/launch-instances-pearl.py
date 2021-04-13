# t4
["asia-east1-c", "australia-southeast1-c", "europe-west2-b", "europe-west4-b", "us-central1-a"]
[4, 4, 4, 4, 4,]

# p100
["australia-southeast1", "asia-east1-a", "europe-west4-a", "us-central1-a", "us-west1-b"]
[4, 4, 4, 4, 4]

# v100
["us-east1-c", "us-central1-a", "europe-west4-a"]
[4, 4, 4]

import subprocess
import os
import click

@click.command()
@click.option('--gpu', default=True, type=bool)
def launch_experiments(gpu):
    # env_names = ["push-v2", "pick-place-v2", "reach-v2"]
    env_names = ["push-v2", "pick-place-v2"]

    instance_groups = [[3, 4, 5], [6, 7, 8], [0, 1, 2],]
    # zones_map = {"v100" : set(["us-central1-a", "asia-east1-c"]),
    #          "p100" : set(["australia-southeast1-c", "asia-east1-a", "europe-west4-a", "us-central1-a", "us-west1-b", ])}
    zones_map = {"t4" : set(["asia-east1-c", "australia-southeast1-c", "europe-west2-b", "europe-west4-b", "us-central1-a"])}
    zones = ["asia-east1-c", "australia-southeast1-c", "europe-west2-b", "europe-west4-b", "us-central1-a"]
    counter = 0
    for env_name in env_names:
        instances = instance_groups.pop()
        for i in range(10):
            if not counter % 4:
                zone = zones.pop(0)
                # if zone in zones_map['v100']:
                #     gpu_type = "v100"
                # elif zone in zones_map['p100']:
                #     gpu_type = "p100"
                # else:
                #     raise ValueError(zone)
                gpu_type = "t4"
            if not counter % 4:
                instance_num = instances.pop(0)
            counter += 1
            ####################EDIT THESE FIELDS##################
            username = f'avnishnarayan' # your google username
            algorithm = f'pearl'
            zone = zone # find the apprpropriate zone here https://cloud.google.com/compute/docs/regions-zones
            instance_name = f'ml1-{env_name}-round3-v2-{algorithm}-{i}'
            bucket = f'ml1/round3/{env_name}/{algorithm}/v2'
            branch = 'avnish-new-metaworld-results-mt1'
            experiment = f'metaworld_launchers/ml1/pearl_metaworld_ml1.py --env_name {env_name}'
            ######################################################

            if not gpu:
                machine_type =  'n2-standard-8' # 'c2-standard-4' we have a quota of 24 of each of these cpus per zone. 
                # You can use n1 cpus which are slower, but we are capped to a total of 72 cpus per zone anyways
                docker_run_file = 'docker_metaworld_run_cpu.py' # 'docker_metaworld_run_gpu.py' for gpu experiment
                docker_build_command = 'make run-dev -C ~/garage/'
                source_machine_image = 'metaworld-v2-cpu-instance'
                launch_command = (f"gcloud beta compute instances create {instance_name} "
                    f"--metadata-from-file startup-script=launchers/launch-experiment-{algorithm}-{i}.sh --zone {zone} "
                    f"--source-machine-image {source_machine_image} --machine-type {machine_type}")
            else:
                machine_type =  'n1-highmem-8'
                docker_run_file = 'docker_metaworld_run_gpu.py'
                docker_build_command = ("make run-dev-nvidia-headless -C ~/garage/ "
                    '''PARENT_IMAGE='nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04' ''')
                source_machine_image = f'gpu-instance-{instance_num}'
                accelerator = f'"type=nvidia-tesla-{gpu_type},count=1"'
                launch_command = (f"gcloud beta compute instances create {instance_name} "
                    f"--metadata-from-file startup-script=launchers/launch-experiment-{algorithm}-{env_name}-{i}.sh --zone {zone} "
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

            with open(f'launchers/launch-experiment-{algorithm}-{env_name}-{i}.sh', mode='w') as f:
                f.write(script)
            subprocess.Popen([launch_command], shell=True)
            print(launch_command)

launch_experiments()
