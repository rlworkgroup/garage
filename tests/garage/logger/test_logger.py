import unittest

from garage.logger import CsvOutput, logger, tabular, TextOutput
from garage.logger.logger import LoggerWarning
from garage.misc.console import remove_if_exists


class TestLogger(unittest.TestCase):
    def setUp(self):
        tabular.clear()

    def tearDown(self):
        logger.remove_all()

    def test_outputs(self):
        tabular.clear()
        with self.assertWarns(LoggerWarning):
            logger.log('test')

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

            tabular.record_misc_stat('stat', 1)
            with self.assertWarns(LoggerWarning):
                logger.log(tabular)
        finally:
            for file in log_files:
                remove_if_exists(file)
            remove_if_exists(csv_file)

    def test_errors(self):
        with self.assertRaises(ValueError):
            logger.add_output(TextOutput)

        with self.assertRaises(ValueError):
            logger.add_output('test_string')

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
