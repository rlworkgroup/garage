import unittest

from garage.logger import Logger, StdOutput, TextOutput
from garage.misc.console import remove_if_exists


class TestTextOutput(unittest.TestCase):
    def setUp(self):
        self.log_file = 'test_text_output.txt'
        self.logger = Logger()
        self.text_output = TextOutput(self.log_file)
        self.logger.add_output(self.text_output)

    def tearDown(self):
        self.logger.remove_all()
        remove_if_exists(self.log_file)


class TestStdOutput(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()
        self.std_output = StdOutput()
        self.logger.add_output(self.std_output)

    def tearDown(self):
        self.logger.remove_all()
