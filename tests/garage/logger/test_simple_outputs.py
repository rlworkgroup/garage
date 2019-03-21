from contextlib import redirect_stdout
import datetime
import io
import tempfile
import unittest
from unittest import mock

from garage.logger import StdOutput, TabularInput, TextOutput

FAKE_TIMESTAMP = '2000-01-01 00:00:00.000000'
FAKE_TIMESTAMP_SHORT = '2000-01-01 00:00:00'


def fake_timestamp(mock_datetime):
    mock_datetime.datetime.now.return_value = datetime.datetime(2000, 1, 1)
    mock_datetime.datetime.return_value.strftime.return_value = FAKE_TIMESTAMP


@mock.patch('garage.logger.simple_outputs.datetime')
class TestTextOutput(unittest.TestCase):
    def setUp(self):
        self.log_file = tempfile.NamedTemporaryFile()
        self.text_output = TextOutput(self.log_file.name)
        self.tabular = TabularInput()

    def tearDown(self):
        self.log_file.close()

    def test_record(self, mock_datetime):
        fake_timestamp(mock_datetime)

        text = 'TESTING 123 GARAGE'
        self.text_output.record(text)
        self.text_output.dump()

        with open(self.log_file.name, 'r') as file:
            correct = '{} | TESTING 123 GARAGE\n'.format(FAKE_TIMESTAMP_SHORT)
            assert file.read() == correct

        more_text = 'MORE TESTING'
        self.text_output.record(more_text)
        self.text_output.dump()

        with open(self.log_file.name, 'r') as file:
            correct = (
                '{} | TESTING 123 GARAGE\n'
                '{} | MORE TESTING\n'
                .format(FAKE_TIMESTAMP_SHORT, FAKE_TIMESTAMP_SHORT)
            )  # yapf: disable
            assert file.read() == correct

    def test_record_no_timestamp(self, mock_datetime):
        fake_timestamp(mock_datetime)
        self.text_output = TextOutput(self.log_file.name, with_timestamp=False)

        text = 'TESTING 123 GARAGE'
        self.text_output.record(text)
        self.text_output.dump()

        with open(self.log_file.name, 'r') as file:
            correct = 'TESTING 123 GARAGE\n'
            assert file.read() == correct

    def test_record_tabular(self, mock_datetime):
        fake_timestamp(mock_datetime)

        self.tabular.record('foo', 100)
        self.tabular.record('bar', 55)

        self.text_output.record(self.tabular)
        self.text_output.dump()

        with open(self.log_file.name, 'r') as file:
            tab = (
                '---  ---\n'
                'bar   55\n'
                'foo  100\n'
                '---  ---\n'
            )  # yapf: disable
            assert file.read() == tab

    def test_record_unknown(self, mock_datetime):
        with self.assertRaises(ValueError):
            self.text_output.record(dict())


@mock.patch('garage.logger.simple_outputs.datetime')
class TestStdOutput(unittest.TestCase):
    def setUp(self):
        self.tabular = TabularInput()
        self.std_output = StdOutput(with_timestamp=False)
        self.str_out = io.StringIO()

    def tearDown(self):
        self.str_out.close()

    def test_record_str(self, mock_datetime):
        fake_timestamp(mock_datetime)

        with redirect_stdout(self.str_out):
            self.std_output.record('test')

        self.std_output.dump()

        self.str_out.seek(0)
        assert self.str_out.read() == 'test\n'

    def test_record_tabular(self, mock_datetime):
        fake_timestamp(mock_datetime)

        self.tabular.record('foo', 100)
        self.tabular.record('bar', 55)

        with redirect_stdout(self.str_out):
            self.std_output.record(self.tabular)

        self.std_output.dump()

        tab = (
            '---  ---\n'
            'bar   55\n'
            'foo  100\n'
            '---  ---\n'
        )  # yapf: disable
        self.str_out.seek(0)
        assert self.str_out.read() == tab

    def test_record_with_timestamp(self, mock_datetime):
        fake_timestamp(mock_datetime)

        self.std_output = StdOutput(with_timestamp=True)

        with redirect_stdout(self.str_out):
            self.std_output.record('garage')

        self.std_output.dump()

        self.str_out.seek(0)
        contents = self.str_out.read()
        assert contents == '{} | garage\n'.format(FAKE_TIMESTAMP_SHORT)

    def test_record_unknown(self, mock_datetime):
        with self.assertRaises(ValueError):
            self.std_output.record(dict())
