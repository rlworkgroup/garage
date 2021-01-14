import argparse
import glob
import os
import time

parser = argparse.ArgumentParser(description='upload experiments directory')
parser.add_argument('subdir', metavar='s', type=str, help='subdirectory of gcs bucket')
parser.add_argument('wait', metavar='t', type=int, help='time to wait between copies in seconds')
args = parser.parse_args()
subdir = args.subdir
wait_time = args.wait

from google.cloud import storage

storage_client = storage.Client()
mw_bucket = storage.Bucket(storage_client, "metaworld-v2-paper-results")


def copy_local_directory_to_gcs(local_path, bucket, gcs_path):
    """Recursively copy a directory of files to GCS.

    local_path should be a directory and not have a trailing slash.
    """
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            copy_local_directory_to_gcs(local_file, bucket, gcs_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
            print(remote_path)
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)

copy_local_directory_to_gcs('../../metaworld-runs-v2/local/experiment', mw_bucket, subdir)

while 1:
    try:
        copy_local_directory_to_gcs('../../metaworld-runs-v2/local/experiment', mw_bucket, subdir)
        
    except:
        print("Wasn't able to upload, trying again in 10 seconds.")
        time.sleep(10)
