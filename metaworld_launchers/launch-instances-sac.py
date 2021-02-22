import os
import subprocess
import time

import click


@click.command()
@click.option('--gpu', default=False, type=bool)
def launch_experiments(gpu):
    env_names = [
        # 'plate-slide-back-side-v2',
        # 'plate-slide-back-v2',
        # 'push-back-v2',
        # 'push-v2',
        # 'push-wall-v2',
        # 'reach-v2',
        # 'reach-wall-v2',
        # 'shelf-place-v2',
        # 'soccer-v2',
        # 'stick-pull-v2',
        # 'stick-push-v2',
        # 'sweep-into-v2',
        # 'sweep-v2',
        # 'dial-turn-v2',
        # 'door-close-v2',
        # 'drawer-close-v2',
        # "faucet-close-v2",
        # "faucet-open-v2",
        # 'handle-press-side-v2',
        # 'handle-press-v2',
        # "handle-pull-side-v2",
        # "handle-pull-v2",
        # 'peg-insert-side-v2',
        # "peg-unplug-side-v2",
        # 'pick-place-v2',
        # 'pick-place-wall-v2',
        # 'plate-slide-side-v2',
        # 'plate-slide-v2',
        # 'window-close-v2',
        # 'window-open-v2',
        # 'assembly-v2',
        # 'basketball-v2',
        'bin-picking-v2',
        # 'box-close-v2',
        # 'button-press-topdown-v2',
        # 'button-press-topdown-wall-v2',
        # 'button-press-v2',
        # 'button-press-wall-v2',
        # 'coffee-button-v2',
        # 'coffee-pull-v2',
        # 'coffee-push-v2',
        # 'disassemble-v2',
        # 'door-lock-v2',
        # 'door-open-v2',
        # 'door-unlock-v2',
        # 'drawer-open-v2',
        # 'hammer-v2',
        # 'hand-insert-v2',
        # 'lever-pull-v2',
        # 'pick-out-of-hole-v2',
    ]
    for i, env_name in enumerate(env_names):
        ####################EDIT THESE FIELDS##################
        username = f'haydenshively'  # your google username
        zone = f'europe-west1-b'  # find the apprpropriate zone here https://cloud.google.com/compute/docs/regions-zones
        instance_name = f'sac-metaworld-{env_name}'
        bucket = f'sac_round2/{env_name}'
        branch = 'hayden-new-metaworld-results-st-v2'
        experiment = f'metaworld_launchers/single_task_launchers/sac_metaworld.py --env_name {env_name}'
        ######################################################

        if not gpu:
            machine_type = 'n1-standard-16'  # 'c2-standard-4' we have a quota of 24 of each of these cpus per zone.
            # You can use n1 cpus which are slower, but we are capped to a total of 72 cpus per zone anyways
            docker_run_file = 'docker_metaworld_run_cpu.py'  # 'docker_metaworld_run_gpu.py' for gpu experiment
            docker_build_command = 'make run-headless -C ~/garage/'
            source_machine_image = 'metaworld-v2-cpu-instance'
            launch_command = (
                f"gcloud beta compute instances create {instance_name} "
                f"--metadata-from-file startup-script=launchers/launch-experiment.sh --zone {zone} "
                f"--source-machine-image {source_machine_image} --machine-type {machine_type}")
        else:
            machine_type = 'n1-standard-4'
            docker_run_file = 'docker_metaworld_run_gpu.py'
            docker_build_command = ("make run-nvidia-headless -C ~/garage/ "
                                    '''PARENT_IMAGE='nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04' ''')
            source_machine_image = 'metaworld-v2-gpu-instance'
            accelerator = '"type=nvidia-tesla-k80,count=1"'
            launch_command = (
                f"gcloud beta compute instances create {instance_name} "
                f"--metadata-from-file startup-script=launchers/launch-experiment.sh --zone {zone} "
                f"--source-machine-image {source_machine_image} --machine-type {machine_type} "
                f'--accelerator={accelerator}')

        os.makedirs('launchers/', exist_ok=True)

        script = (
            '#!/bin/bash\n'
            f'cd /home/{username}\n'
            f'cp -r /home/avnishnarayan/.mujoco .\n'
            f'runuser -l {username} -c "git clone https://github.com/rlworkgroup/garage'
            f' && cd garage/ && git checkout {branch} && mkdir data/"\n'
            f'runuser -l {username} -c "mkdir -p metaworld-runs-v2/local/experiment/"\n'
            f'runuser -l {username} -c "{docker_build_command}"\n'
            f'''runuser -l {username} -c "cd garage && python {docker_run_file} '{experiment}'"\n'''
            f'runuser -l {username} -c "cd garage/metaworld_launchers && python upload_folders.py {bucket} 1200"\n')

        with open(f'launchers/launch-experiment.sh', mode='w') as f:
            f.write(script)
        if i % 3:
            time.sleep(500)
        print(launch_command)
        subprocess.run([launch_command], shell=True)


launch_experiments()
