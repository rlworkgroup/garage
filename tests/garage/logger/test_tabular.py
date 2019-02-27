import csv
import math
import random
import unittest

import numpy as np

from garage.logger import CsvOutput, HistogramInput, logger, tabular
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

            tabular.record_misc_stat("Foo", [0, 1, 2])
            bar = np.random.rand(10)
            tabular.record_misc_stat("Bar", bar.tolist(), placement='front')
            tabular.record_misc_stat("none", None)
            logger.log(tabular)

            kv_dict = {
                'FooAverage': 1.0,
                'FooStd': 0.816496580927726,
                'FooMedian': 1.0,
                'FooMin': 0,
                'FooMax': 2,
                'AverageBar': np.average(bar),
                'StdBar': np.std(bar),
                'MedianBar': np.median(bar),
                'MinBar': np.min(bar),
                'MaxBar': np.max(bar),
                'noneAverage': 'nan',
                'noneStd': 'nan',
                'noneMedian': 'nan',
                'noneMin': 'nan',
                'noneMax': 'nan'
            }
            self.check_csv_first_row(log_file, kv_dict)
        finally:
            remove_if_exists(log_file)

    def test_prefix(self):
        log_file = 'test_tabular.csv'
        try:
            logger.add_output(CsvOutput(log_file))
            tabular.clear()

            foo = random.randint(0, 999)
            bar = random.randint(0, 999)
            with tabular.prefix('test_'):
                tabular.record("foo", foo)
                tabular.record("bar", bar)
            logger.log(tabular)

            kv_dict = {'test_foo': foo, 'test_bar': bar}
            self.check_csv_first_row(log_file, kv_dict)
        finally:
            remove_if_exists(log_file)

    def test_str(self):
        tabular.clear()

        foo = random.randint(0, 999)
        bar = random.randint(0, 999)
        baz = np.random.rand(10).tolist()
        tabular.record("foo", foo)
        tabular.record("bar", bar)
        tabular.record("baz", HistogramInput(baz))

        correct_str = "---  ---\nfoo  {:>3}\nbar  {:>3}\n---  ---"
        correct_str = correct_str.format(foo, bar)
        assert (str(tabular) == correct_str)

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
            if kv_dict[key] == 'nan':
                assert str(value) == kv_dict[key]
            else:
                assert math.isclose(kv_dict[key], float(value))
