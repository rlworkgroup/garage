"""
Auxiliary coverage test for logger.

This test covers snapshot and parameter logging.
"""
import unittest

from garage.logger import logger, StdOutput, tabular


class TestLogger(unittest.TestCase):
    def test_logger(self):
        logger.add_output(StdOutput())
        tabular.record_misc_stat("key", 1)

        logger.log(tabular)
