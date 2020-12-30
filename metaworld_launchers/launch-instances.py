import subprocess

num_experiments = 3
zone = 'us-west2-a'
machine_type = 'c2-standard-8'
source_machine_image = 'metaworld-ml10-mt10-cpu'

for i in range(num_experiments):
    subprocess.run([f"gcloud beta compute instances create rl2-ml1-push-{i} --metadata-from-file startup-script=launch-experiment.sh --zone {zone} --source-machine-image {source_machine_image} --machine-type {machine_type}"], shell=True)
