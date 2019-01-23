"""
Auxiliary coverage test for logger.

This test covers snapshot and parameter logging.
"""
import os
from types import SimpleNamespace
import unittest

from garage.logger import logger, snapshotter, StdOutput, tabular


class TestLogger(unittest.TestCase):
    def test_logger(self):
        args = SimpleNamespace()
        args.args_data = ""

        logger.add_output(StdOutput())
        snapshotter.set_snapshot_dir("exp")
        snapshotter.save_itr_params(1, {})
        tabular.record_misc_stat("key", 1)
        if os.path.isfile("exp-log"):
            os.remove("exp-log")
        logger.log(tabular)
