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


# Function get_gcs_bucket also works to verify that the certificate and bucket
# name are correct, which is useful before running an experiment to avoid a
# late error message once the training has finished.
check_gcs_config = get_gcs_bucket


def upload_to_gcs(files_to_upload_path: str,
                  bucket: str,
                  path_in_bucket: str = ""):
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
