import argparse
import json
import os
import subprocess
import uuid

import joblib

from garage.algos import BatchPolopt
from garage.misc import logger
from garage.misc.instrument import to_local_command

filename = str(uuid.uuid4())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=None,
        help='path to the new log directory')
    # Look for params.json file
    args = parser.parse_args()
    parent_dir = os.path.dirname(os.path.realpath(args.file))
    json_file_path = os.path.join(parent_dir, "params.json")
    logger.log("Looking for params.json at %s..." % json_file_path)
    try:
        with open(json_file_path, "r") as f:
            params = json.load(f)
        # exclude certain parameters
        excluded = ['json_args']
        for k in excluded:
            if k in params:
                del params[k]
        for k, v in list(params.items()):
            if v is None:
                del params[k]
        if args.log_dir is not None:
            params['log_dir'] = args.log_dir
        params['resume_from'] = args.file
        command = to_local_command(params, script='scripts/run_experiment.py')
        print(command)
        try:
            subprocess.call(command, shell=True, env=os.environ)
        except Exception as e:
            print(e)
            if isinstance(e, KeyboardInterrupt):
                raise
    except IOError as e:
        logger.log("Failed to find json file. Continuing in non-stub mode...")
        data = joblib.load(args.file)
        assert 'algo' in data
        algo = data['algo']
        assert isinstance(algo, BatchPolopt)
        algo.train()
