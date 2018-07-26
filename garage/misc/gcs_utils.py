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

from google.cloud import storage
from termcolor import colored

from garage.config import GCS_CREDENTIAL_PATH


def get_gcs_bucket(bucket: str):
    """Obtain the GCS bucket.

    It's verified that the JSON credentials to connect to GCS is set, and then
    the bucket is retrieved.

    Parameters
    ----------
    bucket: str
        The bucket name as defined in GCS.

    """
    # Name of the environment variable used by the GCS library
    GCS_ENV_VAR = "GOOGLE_APPLICATION_CREDENTIALS"
    if GCS_ENV_VAR not in os.environ:
        assert os.path.isfile(GCS_CREDENTIAL_PATH), colored(
            "No credentials for GCS were provided. For instructions on " \
            "how to setup credentials, see \"Set up a service acccount\":\n" \
            "https://cloud.google.com/video-intelligence/docs/common"\
            "/auth\n" \
            "Besides an environment variable, the certificate path can " \
            "be set in field GCS_CREDENTIAL_PATH at config_personal.py.",
            "yellow")
        os.environ[GCS_ENV_VAR] = GCS_CREDENTIAL_PATH
    storage_client = storage.Client()
    return storage_client.get_bucket(bucket)

"""
Function get_gcs_bucket also works to verify that the certificate and bucket
name are correct, which is useful before running an experiment to avoid a
late error message once the training has finished.
"""
check_gcs_config = get_gcs_bucket


def upload_to_gcs(files_to_upload_path: str,
                  bucket: str,
                  path_in_bucket: str = ""):
    """Upload files under the specified bucket.

    All the files under the specified path will uploaded to GCS. Only the last
    folder in the path will be added to GCS. For example, if the following
    path is passed:
        "/home/user/Documents/project/files_to_upload/"
    Only the folder "files_to_upload_path" will be added to GCS in order to
    contain all the files in it.
    However, a specific path can be prepended to the added folder relative to
    the root of the bucker. Such path can be set in parameter "path_in_bucket".

    Parameters
    ----------
    files_to_upload_path: str
        Specify the path of the file or files to upload to GCS.
    bucket: str
        Name of the bucket in your GCS account.
    path_in_bucket:
        path prepended to the root folder or file uploaded to GCS.

    """
    bucket = get_gcs_bucket(bucket)
    path_in_bucket = path_in_bucket.rstrip("/")
    base_dir = files_to_upload_path.rstrip("/").rsplit("/", 1)[0]
    if os.path.isfile(files_to_upload_path):
        gcs_filename = files_to_upload_path.replace(base_dir, "")
        blob = bucket.blob(path_in_bucket + gcs_filename)
        blob.upload_from_filename(files_to_upload_path)
    elif os.path.isdir(files_to_upload_path):
        for filename in glob.iglob(
                files_to_upload_path + "/**", recursive=True):
            if os.path.isfile(filename):
                gcs_filename = filename.replace(base_dir, "")
                blob = bucket.blob(path_in_bucket + gcs_filename)
                blob.upload_from_filename(filename)
