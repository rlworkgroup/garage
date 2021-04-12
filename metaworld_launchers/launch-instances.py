import subprocess
import os
import click

@click.command()
@click.option('--gpu', default=False, type=bool)
def launch_experiments(gpu):
    learning_rates = [7e-5, 3e-5, 1e-6, 7e-6, 5e-6, 3e-6]
    for i, inner_learning_rate in enumerate(learning_rates):
        ####################EDIT THESE FIELDS##################
        username = f'avnishnarayan' # your google username
        algorithm = f'maml-trpo'
        zone = f'us-west1-a' # find the apprpropriate zone here https://cloud.google.com/compute/docs/regions-zones
        entropy_str = str(inner_learning_rate).replace('.', '-')
        instance_name = f'maml-ml10-tuned-inner-learning-rate-{entropy_str}'
        bucket = f'ml10/maml-tuned'
        branch = 'run-maml-linear-baseline'
        experiment = f'metaworld_launchers/ml10/maml_trpo_metaworld_ml10.py --inner_lr {inner_learning_rate}'
        ######################################################

        if not gpu:
            machine_type =  'n2-standard-8' # 'c2-standard-4' we have a quota of 24 of each of these cpus per zone. 
            # You can use n1 cpus which are slower, but we are capped to a total of 72 cpus per zone anyways
            docker_run_file = 'docker_metaworld_run_cpu.py' # 'docker_metaworld_run_gpu.py' for gpu experiment
            docker_build_command = 'make run-dev -C ~/garage/'
            source_machine_image = 'metaworld-v2-cpu-instance'
            launch_command = (f"gcloud beta compute instances create {instance_name} "
                f"--metadata-from-file startup-script=launchers/launch-experiment-{i}.sh --zone {zone} "
                f"--source-machine-image {source_machine_image} --machine-type {machine_type}")
        else:
            machine_type =  'n1-standard-4' 
            docker_run_file = 'docker_metaworld_run_gpu.py'
            docker_build_command = ("make run-dev-nvidia-headless -C ~/garage/ "
                '''PARENT_IMAGE='nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04' ''')
            source_machine_image = 'metaworld-v2-gpu-instance'
            accelerator = '"type=nvidia-tesla-k80,count=1"'
            launch_command = (f"gcloud beta compute instances create {instance_name} "
                f"--metadata-from-file startup-script=launchers/launch-experiment-{i}.sh --zone {zone} "
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

        with open(f'launchers/launch-experiment-{i}.sh', mode='w') as f:
            f.write(script)

        subprocess.Popen([launch_command], shell=True)
        print(launch_command)

launch_experiments()
