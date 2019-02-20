"""
Auxiliary coverage test for logger.

This test covers snapshot and parameter logging.
"""
import os
import unittest

from garage.logger import CsvOutput, logger, tabular, TextOutput


class TestLogger(unittest.TestCase):
    def test_logger(self):
        log_file = 'text_logger.txt'
        try:
            text_output = TextOutput(log_file)
            logger.add_output(text_output)

            text = "TESTING 123 GARAGE"
            logger.log(text)
            with open(log_file, 'r') as file:
                read = file.read().split(text_output._delimiter, 1)[1]
                read = read.rstrip('\n')
                assert read == text

            more_text = "MORE TESTING"
            logger.log(more_text)
            with open(log_file, 'r') as file:
                read = file.read().split('\n')
                for i in range(len(read) - 1):
                    read[i] = read[i].split(text_output._delimiter, 1)[1]
                assert read[0] == text
                assert read[1] == more_text
        finally:
            os.remove(log_file)

    def test_tabular(self):
        logger.add_output(CsvOutput())
        tabular.record_misc_stat("key", 1)

        logger.log(tabular)
