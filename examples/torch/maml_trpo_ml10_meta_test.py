#!/usr/bin/env python3
"""This is an example to do meta-testing on ML10 environment."""
from metaworld.benchmarks import ML10

from garage.experiment.offline_meta_test_helper \
    import OfflineTorchMetaTestHelper

if __name__ == '__main__':
    # Run this script by `python maml_trpo_ml10_meta_test.py folder`
    # where folder is an experiment folder that saves snapshot files.
    OfflineTorchMetaTestHelper.read_cmd(ML10.get_test_tasks)
