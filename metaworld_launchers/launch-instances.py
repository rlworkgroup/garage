import subprocess

algorithm ='maml-trpo'
experiment_type = 'ml10-rerun'
num_experiments = 3
zone = 'us-west1-a'
machine_type = 'c2-standard-8'
source_machine_image = 'metaworld-ml10-mt10-cpu'


for i in range(num_experiments):
    subprocess.run([f"gcloud beta compute instances create {algorithm}-{experiment_type}-{i} --metadata-from-file startup-script=launch-experiment.sh --zone {zone} --source-machine-image {source_machine_image} --machine-type {machine_type}"], shell=True)
