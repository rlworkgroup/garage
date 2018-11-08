import os
import garage.torch

# Change this as desired
LOCAL_LOG_DIR = 'output'

CODE_DIRS_TO_MOUNT = [
    os.path.dirname(os.path.dirname(garage.torch.__file__))
    # '/path/to/other/git/repos/',
]
