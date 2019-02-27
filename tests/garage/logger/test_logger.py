import unittest

from garage.logger import CsvOutput, logger, tabular, TextOutput
from garage.misc.console import remove_if_exists


class TestLogger(unittest.TestCase):
    def test_logger(self):
        logger.disable_warnings()
        assert logger.log(
            "test") == "No outputs have been added to the logger."

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
            logger.remove_all()
            remove_if_exists(log_file)

    def test_outputs(self):
        log_files = ['test_%u.txt' % i for i in range(5)]
        csv_file = 'text.csv'
        try:
            logger.disable_warnings()

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
            logger.remove_all()
            for file in log_files:
                remove_if_exists(file)
            remove_if_exists(csv_file)

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
            logger.remove_all()
            remove_if_exists(log_file)
            remove_if_exists(csv_file)
