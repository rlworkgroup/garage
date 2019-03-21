import unittest

from garage.logger import Logger, LogOutput
from garage.logger.logger import LoggerWarning


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.mock_output = unittest.mock.Mock(
            spec=LogOutput, types_accepted=(str, ))
        self.mock_output_type = type(self.mock_output)
        self.logger = Logger()

    def test_add_output(self):
        self.logger.add_output(self.mock_output)
        assert self.logger.has_output_type(self.mock_output_type)

    def test_add_uninstantiated_output(self):
        with self.assertRaises(ValueError):
            self.logger.add_output(self.mock_output_type)

    def test_add_unknown_output(self):
        with self.assertRaises(ValueError):
            self.logger.add_output('foo')

    def test_log(self):
        self.logger.add_output(self.mock_output)
        self.logger.log('foo')
        self.mock_output.record.assert_called_with('foo', prefix='')

    def test_log_with_no_output(self):
        with self.assertWarns(LoggerWarning):
            self.logger.log('foo')

    def test_log_with_no_logged(self):
        with self.assertWarns(LoggerWarning):
            self.logger.add_output(self.mock_output)
            self.logger.log(dict())

    def test_log_with_prefix(self):
        with self.logger.prefix('a/'):
            self.logger.add_output(self.mock_output)
            self.logger.log('foo')
            self.mock_output.record.assert_called_with('foo', prefix='a/')

    def test_remove_output_type(self):
        self.logger.add_output(self.mock_output)
        self.logger.remove_output_type(self.mock_output_type)
        assert not self.logger.has_output_type(self.mock_output_type)

    def test_remove_all(self):
        self.logger.add_output(self.mock_output)
        self.logger.remove_all()
        assert not self.logger._outputs

    def test_reset_output(self):
        self.logger.add_output(self.mock_output)
        self.logger.reset_output(self.mock_output)
        assert self.logger.has_output_type(self.mock_output_type)

    def test_dump_output_type(self):
        self.logger.add_output(self.mock_output)
        self.logger.dump_output_type(self.mock_output_type)
        self.mock_output.dump.assert_called_with(step=None)

    def test_dump_all(self):
        self.logger.add_output(self.mock_output)
        self.logger.dump_all()
        self.mock_output.dump.assert_called_with(step=None)
