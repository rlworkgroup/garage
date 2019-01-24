"""Contains the output classes for the logger.

Each class is sent logger data and handles it itself.
"""
import abc
import csv
import datetime
import os
import sys

import dateutil.tz

from garage.logger.tabular_input import TabularInput
from garage.misc.console import mkdir_p


class LogOutput(abc.ABC):
    """Abstract class for Logger Outputs."""

    @property
    def types_accepted(self):
        """The types that the logger may pass to this output."""
        return ()

    @abc.abstractmethod
    def record(self, data, prefix=''):
        """This method is called by the logger when it needs to pass data."""
        pass

    def dump(self, step=None):
        """This method is called by the logger to dump an output."""
        pass


class StdOutput(LogOutput):
    """Standard console output for the logger.

    :param with_timestamp: Whether to log a timestamp before non-tabular data.
    """

    def __init__(self, with_timestamp=True):
        self._with_timestamp = with_timestamp

    @property
    def types_accepted(self):
        """The types that the logger may pass to this output."""
        return (str, TabularInput)

    def record(self, data, prefix=''):
        """Log data to console."""
        if isinstance(data, str):
            out = prefix + data
            if self._with_timestamp:
                now = datetime.datetime.now(dateutil.tz.tzlocal())
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
                out = "%s | %s" % (timestamp, out)
        elif isinstance(data, TabularInput):
            out = str(data)
        else:
            return

        print(out)
        sys.stdout.flush()


class TextOutput(LogOutput):
    """Text file output for logger.

    :param file_name: The file this output should log to.
    :param with_timestamp: Whether to log a timestamp before the data.
    """

    def __init__(self, file_name, with_timestamp=True):
        mkdir_p(os.path.dirname(file_name))
        self._log_file = open(file_name, 'a')
        self._with_timestamp = with_timestamp

    @property
    def types_accepted(self):
        """The types that the logger may pass to this output."""
        return (str, )  # we need a comma here so the return value is a tuple

    def record(self, data, prefix=''):
        """Log data to text file."""
        out = data
        if self._with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)

        self._log_file.write(out + '\n')
        self._log_file.flush()


class CsvOutput(LogOutput):
    """CSV file output for logger.

    :param file_name: The file this output should log to.
    """

    def __init__(self, file_name):
        mkdir_p(os.path.dirname(file_name))
        self._log_file = open(file_name, 'w')

        self._tabular_header_written = False

    @property
    def types_accepted(self):
        """The types that the logger may pass to this output."""
        return (TabularInput, )  # we need the comma here so the return value
        # is a tuple

    def record(self, data, prefix=''):
        """Log tabular data to CSV."""
        writer = csv.DictWriter(
            self._log_file, fieldnames=data.get_table_key_set())

        if not self._tabular_header_written:
            writer.writeheader()
            self._tabular_header_written = True
        writer.writerow(data.get_table_dict())
        self._log_file.flush()
