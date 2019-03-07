from contextlib import redirect_stdout
import datetime
import io
import unittest

import mock

from garage.logger import CsvOutput, logger, StdOutput, tabular, TextOutput
from garage.misc.console import remove_if_exists


class TestLogger(unittest.TestCase):
    def setUp(self):
        super().setUp()
        logger.disable_warnings()

    def tearDown(self):
        super().tearDown()
        logger.remove_all()

    def test_std(self):
        tabular.clear()
        logger.add_output(StdOutput(with_timestamp=False))

        with io.StringIO() as str_out:
            with redirect_stdout(str_out):
                logger.log("test")
            str_out.seek(0)
            assert str_out.read() == "test\n"

        tabular.record("foo", 100)
        tabular.record("bar", 55)

        with io.StringIO() as str_out:
            with redirect_stdout(str_out):
                logger.log(tabular)
            tab = "---  ---\nfoo  100\nbar   55\n---  ---\n"
            str_out.seek(0)
            assert str_out.read() == tab

    @mock.patch('garage.logger.outputs.datetime')
    def test_timestamp(self, mock_datetime):
        timestamp = '2000-01-01 00:00:00'
        mock_datetime.datetime.now.return_value = datetime.datetime(2000, 1, 1)
        mock_datetime.datetime.return_value.strftime.return_value = timestamp

        logger.reset_output(StdOutput())

        with io.StringIO() as str_out:
            with redirect_stdout(str_out):
                logger.log("garage")
            str_out.seek(0)
            assert str_out.read() == timestamp + ' | ' + 'garage\n'

    def test_text(self):
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
            remove_if_exists(log_file)

    def test_outputs(self):
        tabular.clear()
        err = "No outputs have been added to the logger."
        assert logger.log("test") == err

        log_files = ['test_%u.txt' % i for i in range(5)]
        csv_file = 'text.csv'
        try:
            logger.add_output(CsvOutput(csv_file))
            for file in log_files:
                logger.add_output(TextOutput(file))

            assert logger.has_output_type(CsvOutput)
            assert logger.has_output_type(TextOutput)
            logger.remove_output_type(CsvOutput)
            assert not logger.has_output_type(CsvOutput)
            assert logger.has_output_type(TextOutput)

            tabular.record_misc_stat("stat", 1)
            warn = "Log data of type " + type(tabular).__name__
            warn += " was not accepted by any output"
            assert logger.log(tabular) == warn
        finally:
            for file in log_files:
                remove_if_exists(file)
            remove_if_exists(csv_file)

    def test_errors(self):
        with self.assertRaises(ValueError):
            logger.add_output(TextOutput)

        with self.assertRaises(ValueError):
            logger.add_output("test_string")

    def test_add_remove_outputs(self):
        log_file = 'test.txt'
        log_files = ['test_%u.txt' % i for i in range(5)]
        csv_file = 'test.csv'
        try:
            logger.add_output(CsvOutput(csv_file))
            logger.add_output(TextOutput(csv_file))

            assert logger.has_output_type(CsvOutput)
            assert logger.has_output_type(TextOutput)
            logger.remove_output_type(CsvOutput)
            assert not logger.has_output_type(CsvOutput)
            assert logger.has_output_type(TextOutput)

            logger.add_output(CsvOutput(csv_file))
            assert logger.has_output_type(CsvOutput)
            assert logger.has_output_type(TextOutput)
            logger.remove_all()
            assert not logger.has_output_type(CsvOutput)
            assert not logger.has_output_type(TextOutput)

            for file in log_files:
                logger.add_output(TextOutput(file))
            assert logger.has_output_type(TextOutput)
            assert len(logger._outputs) == len(log_files)
            logger.reset_output(TextOutput(log_file))
            assert logger.has_output_type(TextOutput)
            assert len(logger._outputs) == 1
        finally:
            remove_if_exists(log_file)
            remove_if_exists(csv_file)
            for f in log_files:
                remove_if_exists(f)
