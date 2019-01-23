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
from garage.misc.console import colorize, mkdir_p


class LogOutput(abc.ABC):
    """Abstract class for Logger Outputs."""

    @property
    def types_accepted(self):
        """The types that the logger may pass to this output."""
        return ()

    @abc.abstractmethod
    def log_output(self, data, **kwargs):
        """This method is called by the logger when it needs to pass data."""
        pass

    def dump(self, step=None):
        """This method is called by the logger to dump an output."""
        pass


class StdOutput(LogOutput):
    """Standard console output for the logger."""

    @property
    def types_accepted(self):
        """The types that the logger may pass to this output."""
        return str, TabularInput

    def log_output(self,
                   data,
                   prefix='',
                   with_timestamp=True,
                   color=None,
                   **kwargs):
        """Log data to console."""
        if isinstance(data, str):
            out = prefix + data
            if with_timestamp:
                now = datetime.datetime.now(dateutil.tz.tzlocal())
                timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
                out = "%s | %s" % (timestamp, out)
            if color is not None:
                out = colorize(out, color)
        elif isinstance(data, TabularInput):
            out = str(data)
        else:
            return

        print(out)
        sys.stdout.flush()


class TextOutput(LogOutput):
    """Text file output for logger."""

    def __init__(self, file_name):
        mkdir_p(os.path.dirname(file_name))
        self._log_file = open(file_name, 'a')

    @property
    def types_accepted(self):
        """The types that the logger may pass to this output."""
        return str,  # we need the comma here so the return value is iterable

    def log_output(self, data, with_timestamp=True, **kwargs):
        """Log data to text file."""
        if not isinstance(data, self.types_accepted):
            return

        out = data
        if with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s | %s" % (timestamp, out)

        self._log_file.write(out + '\n')
        self._log_file.flush()


class CsvOutput(LogOutput):
    """CSV file output for logger."""

    def __init__(self, file_name):
        mkdir_p(os.path.dirname(file_name))
        self._log_file = open(file_name, 'w')

        self._tabular_header_written = False

    @property
    def types_accepted(self):
        """The types that the logger may pass to this output."""
        return TabularInput,  # we need the comma here so the return value
        # is iterable

    def log_output(self, data, prefix='', **kwargs):
        """Log tabular data to CSV."""
        if not isinstance(data, self.types_accepted):
            return

        writer = csv.DictWriter(
            self._log_file, fieldnames=data.get_table_key_set())

        if not self._tabular_header_written:
            writer.writeheader()
            self._tabular_header_written = True
        writer.writerow(data.get_table_dict())
        self._log_file.flush()
