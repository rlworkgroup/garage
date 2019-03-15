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

    def test_record(self, mock_datetime):
        fake_timestamp(mock_datetime)

        text = 'TESTING 123 GARAGE'
        self.text_output.record(text)
        with open(self.log_file.name, 'r') as file:
            correct = '{}  | TESTING 123 GARAGE\n'.format(FAKE_TIMESTAMP)
            assert file.read() == correct

        more_text = 'MORE TESTING'
        self.text_output.record(more_text)

        with open(self.log_file.name, 'r') as file:
            correct = (
                '{}  | TESTING 123 GARAGE\n'
                '{}  | MORE TESTING\n'
                .format(FAKE_TIMESTAMP, FAKE_TIMESTAMP)
            )  # yapf: disable
            assert file.read() == correct


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

        self.str_out.seek(0)
        assert self.str_out.read() == 'test\n'

    def test_record_tabular(self, mock_datetime):
        fake_timestamp(mock_datetime)

        self.tabular.record('foo', 100)
        self.tabular.record('bar', 55)

        with redirect_stdout(self.str_out):
            self.std_output.record(self.tabular)

        tab = (
            '---  ---\n'
            'foo  100\n'
            'bar   55\n'
            '---  ---\n'
        )  # yapf: disable
        self.str_out.seek(0)
        assert self.str_out.read() == tab

    def test_record_with_timestamp(self, mock_datetime):
        fake_timestamp(mock_datetime)

        self.std_output = StdOutput(with_timestamp=True)

        with redirect_stdout(self.str_out):
            self.std_output.record('garage')

        self.str_out.seek(0)
        contents = self.str_out.read()
        assert contents == '{} | garage\n'.format(FAKE_TIMESTAMP_SHORT)
