import csv
import math
import random
import unittest

from garage.logger import CsvOutput, logger, tabular
from garage.misc.console import remove_if_exists


class TestTabular(unittest.TestCase):
    def test_tabular_record(self):
        log_file = 'test_tabular.csv'
        try:
            logger.add_output(CsvOutput(log_file))
            tabular.clear()

            foo = random.randint(0, 999)
            bar = random.randint(0, 999)
            tabular.record("foo", foo)
            tabular.record("bar", bar)
            logger.log(tabular)

            kv_dict = {'foo': foo, 'bar': bar}
            self.check_csv_first_row(log_file, kv_dict)
        finally:
            remove_if_exists(log_file)

    def test_tabular_misc_stat(self):
        log_file = 'test_tabular.csv'
        try:
            logger.add_output(CsvOutput(log_file))

            tabular.clear()
            tabular.record_misc_stat("key", [0, 1, 2])
            logger.log(tabular)

            kv_dict = {
                'keyAverage': 1.0,
                'keyStd': 0.816496580927726,
                'keyMedian': 1.0,
                'keyMin': 0,
                'keyMax': 2
            }
            self.check_csv_first_row(log_file, kv_dict)
        finally:
            remove_if_exists(log_file)

    def check_csv_first_row(self, log_file, kv_dict):
        """Check the first row of a csv file and compare it to known values."""
        with open(log_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            row = next(reader)

            for key in header:
                assert key in kv_dict.keys()
            for key in kv_dict.keys():
                assert key in header
            self.check_csv_row(header, row, kv_dict)

    def check_csv_row(self, header, row, kv_dict):
        """Check a row of a csv file and compare it to known values."""
        for key, value in zip(header, row):
            assert key in kv_dict.keys()
            assert math.isclose(kv_dict[key], float(value))
