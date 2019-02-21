import csv
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
        log_file = 'test_tabular.csv'
        try:
            logger.add_output(CsvOutput(log_file))

            tabular.clear()
            tabular.record_misc_stat("key", 1)
            logger.log(tabular)

            with open(log_file, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)
                row = next(reader)
                for key, value in zip(header, row):
                    assert key in ['keyAverage', 'keyStd', 'keyMedian']
                    value = float(value)
                    if key == 'keyAverage':
                        assert value == 1.0
                    elif key == 'keyStd':
                        assert value == 0
                    elif key == 'keyMedian':
                        assert value == 1.0
        finally:
            os.remove(log_file)

    def test_outputs(self):
        log_files = ['test_%u.txt' % i for i in range(5)]
        csv_output = 'text.csv'
        try:
            logger.disable_warnings()

            logger.add_output(CsvOutput(csv_output))
            for file in log_files:
                logger.add_output(TextOutput(file))

            logger.remove_output_type(CsvOutput)

            tabular.record_misc_stat("stat", 1)
            warn = "Log data of type " + type(tabular).__name__
            warn += " was not accepted by any output"
            assert logger.log(tabular) == warn

            for file in log_files:
                logger.add_output(TextOutput(file))

            logger.remove_all()

        finally:
            for file in log_files:
                os.remove(file)
            os.remove(csv_output)
