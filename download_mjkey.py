import pathlib
import google.cloud.storage as gcs

client = gcs.Client()

#set target file to write to
target = pathlib.Path("mjkey_practice.txt")

#set file to download
FULL_FILE_PATH = "gs://metaworld-v2-paper-results/mjkey.txt"

#open filestream with write permissions
with target.open(mode="wb") as downloaded_file:
    client.download_blob_to_file(FULL_FILE_PATH, downloaded_file)
