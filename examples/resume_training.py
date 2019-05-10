#!/usr/bin/env python3

import argparse
import os
import subprocess

from garage.experiment import to_local_command

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--snapshot_dir',
        type=str,
        default=None,
        help='Directory of the pickle file to resume experiment from.')
    parser.add_argument(
        '--resume_epoch',
        type=str,
        default=None,
        help='Index of epoch to restore from. '
        'Can be "first", "last" or a number. '
        'Not applicable when snapshot_mode="last"')

    args = parser.parse_args()
    params = dict()
    params['resume_from_dir'] = args.snapshot_dir
    if args.resume_epoch is not None:
        params['resume_epoch'] = args.resume_epoch
    command = to_local_command(
        params, script='garage.experiment.experiment_wrapper')
    print(command)
    try:
        subprocess.call(command, shell=True, env=os.environ)
    except Exception as e:
        print(e)
        if isinstance(e, KeyboardInterrupt):
            raise
