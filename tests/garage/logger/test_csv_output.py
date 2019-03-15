import csv
import tempfile
import unittest

from garage.logger import CsvOutput, TabularInput
from garage.logger.csv_output import CsvOutputWarning


class TestCsvOutput(unittest.TestCase):
    def setUp(self):
        self.log_file = tempfile.NamedTemporaryFile()
        self.csv_output = CsvOutput(self.log_file.name)
        self.tabular = TabularInput()
        self.tabular.clear()

    def test_record(self):
        foo = 1
        bar = 10
        self.tabular.record("foo", foo)
        self.tabular.record("bar", bar)
        self.csv_output.record(self.tabular)
        self.tabular.record("foo", foo * 2)
        self.tabular.record("bar", bar * 2)
        self.csv_output.record(self.tabular)

        correct = [
            {'foo': str(foo), 'bar': str(bar)},
            {'foo': str(foo * 2), 'bar': str(bar * 2)},
        ]  # yapf: disable
        self.assert_csv_matches(correct)

    def test_record_inconsistent(self):
        foo = 1
        bar = 10
        self.tabular.record("foo", foo)
        self.csv_output.record(self.tabular)
        self.tabular.record("foo", foo * 2)
        self.tabular.record("bar", bar * 2)

        with self.assertWarns(CsvOutputWarning):
            self.csv_output.record(self.tabular)

        # this should not produce a warning, because we only warn once
        self.csv_output.record(self.tabular)

        correct = [
            {'foo': str(foo)},
            {'foo': str(foo * 2)},
        ]  # yapf: disable
        self.assert_csv_matches(correct)

    def test_disable_warnings(self):
        foo = 1
        bar = 10
        self.tabular.record("foo", foo)
        self.csv_output.record(self.tabular)
        self.tabular.record("foo", foo * 2)
        self.tabular.record("bar", bar * 2)

        self.csv_output.disable_warnings()

        # this should not produce a warning, because we disabled warnings
        self.csv_output.record(self.tabular)

    def assert_csv_matches(self, correct):
        """Check the first row of a csv file and compare it to known values."""
        with open(self.log_file.name, 'r') as file:
            reader = csv.DictReader(file)

            for correct_row in correct:
                row = next(reader)
                self.assertDictEqual(row, correct_row)
