"""
Auxiliary coverage test for logger.

This test covers snapshot and parameter logging.
"""
from types import SimpleNamespace
import unittest

from garage.misc import logger


class TestLogger(unittest.TestCase):
    def test_logger(self):
        args = SimpleNamespace()
        args.args_data = ""

        logger.set_snapshot_dir("experiment")
        logger.save_itr_params(1, {})
        logger.log_parameters_lite("experiment-log", args)
        logger.log_variant("experiment-log", {})
        logger.record_tabular_misc_stat("key", 1)
