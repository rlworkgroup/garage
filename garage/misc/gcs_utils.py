"""Utility functions to upload files to Google Cloud Storage.

Users can upload files to a specific bucket in their GCS account.
Before attempting to upload any file, it's important to obtain an
authentication certificate and set the corresponding environment variable
as mentioned in section "Set up a service account" in the following link:
    https://cloud.google.com/video-intelligence/docs/common/auth
It's also possible to set the value of the environment variable in the field
GCS_CREDENTIAL_PATH in file config_personal.py.
"""

import glob
import os

import google.api_core.exceptions
from google.cloud import storage
from termcolor import colored

from garage.config import GCS_BUCKET
from garage.config import GCS_CREDENTIAL_PATH
from garage.config import GCS_PATH_IN_BUCKET


def get_gcs_bucket(bucket_name: str = GCS_BUCKET):
    """Obtain the GCS bucket.

    It's verified that the JSON credentials to connect to GCS is set, and then
    the bucket is retrieved.

    Parameters
    ----------
    bucket_name: str
        The bucket name as defined in GCS.

    """
    # Name of the environment variable used by the GCS library
    gcs_env_var = "GOOGLE_APPLICATION_CREDENTIALS"
    if gcs_env_var not in os.environ:
        assert os.path.isfile(GCS_CREDENTIAL_PATH), colored(
            "No credentials for GCS were provided. For instructions on " \
            "how to setup credentials, see \"Set up a service acccount\":\n" \
            "https://cloud.google.com/video-intelligence/docs/common"\
            "/auth\n" \
            "Besides an environment variable, the certificate path can " \
            "be set in field GCS_CREDENTIAL_PATH at config_personal.py.",
            "yellow")
        os.environ[gcs_env_var] = GCS_CREDENTIAL_PATH
    storage_client = storage.Client()
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except google.api_core.exceptions.NotFound:
        print(colored("The bucket %s was not be found. The name could be " \
                      "invalid." % bucket_name, "yellow"))
        raise
    return bucket


# Function get_gcs_bucket also works to verify that the certificate and bucket
# name are correct, which is useful before running an experiment to avoid a
# late error message once the training has finished.
check_gcs_config = get_gcs_bucket


def upload_to_gcs(files_to_upload_path: str,
                  bucket_name: str = GCS_BUCKET,
                  path_in_bucket: str = GCS_PATH_IN_BUCKET):
    """Upload files under the specified bucket.

    The directories in the path where the files are uploaded from are not
    copied in the bucket. However, a path to contain the files in the bucket
    can be passed as a parameter to this function.

    Parameters
    ----------
    files_to_upload_path: str
        Specify the path of the file or files to upload to GCS.
    bucket_name: str
        Name of the bucket in your GCS account.
    path_in_bucket:
        path prepended to the root folder or file uploaded to GCS.

    """
    bucket = get_gcs_bucket(bucket_name)
    path_in_bucket = path_in_bucket.strip("/")
    files_to_upload_path = files_to_upload_path.rstrip("/")
    if os.path.isfile(files_to_upload_path):
        # Leave only the file name append it to the path in the bucket
        gcs_filename = files_to_upload_path.strip("/").rsplit("/", 1)[-1]
        blob = bucket.blob(path_in_bucket + "/" + gcs_filename)
        blob.upload_from_filename(files_to_upload_path)
    elif os.path.isdir(files_to_upload_path):
        for filename in glob.iglob(
                files_to_upload_path + "/**", recursive=True):
            if os.path.isfile(filename):
                gcs_filename = filename.replace(files_to_upload_path, "")
                blob = bucket.blob(path_in_bucket + gcs_filename)
                blob.upload_from_filename(filename)
