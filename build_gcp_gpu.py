from google.cloud import storage

import subprocess
import os
from os.path import expanduser


bucket_name = "metaworld-v2-paper-results"
source_blob_name = "mjkey.txt"
home = expanduser("~")
destination_file_name = f"{home}/.mujoco/mjkey.txt"
if not os.path.exists(f'{home}/.mujoco/'):
    os.makedirs(f'{home}/.mujoco/')

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(source_blob_name)
blob.download_to_filename(destination_file_name)

print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))
subprocess.run(['make', 'run-nvidia-headless', 'PARENT_IMAGE="nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04"'])
