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
        """Pass these types to this logger output.

        The types in this tuple will be accepted by this output.

        :return: A tuple containing all valid input types.
        """
        return ()

    @abc.abstractmethod
    def record(self, data, prefix=''):
        """Pass logger data to this output.

        :param data: The data to be logged by the output.
        :param prefix: A prefix placed before a log entry in text outputs.
        """
        pass

    def dump(self, step=None):
        """Dump the contents of this output.

        :param step: The current run step.
        """
        pass

    def close(self):
        """Close any files used by the output."""
        pass


class NullOutput(LogOutput):
    """Dummy output to disable 'no logger output' warnings."""

    @property
    def types_accepted(self):
        """Accept all output types."""
        return (object, )

    def record(self, data, prefix=''):
        """Don't do anything."""
        pass


class StdOutput(LogOutput):
    """Standard console output for the logger.

    :param with_timestamp: Whether to log a timestamp before non-tabular data.
    """

    def __init__(self, with_timestamp=True):
        self._with_timestamp = with_timestamp

    @property
    def types_accepted(self):
        """Accept str and TabularInput objects."""
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
        self._delimiter = " | "

    @property
    def types_accepted(self):
        """Accept str objects only."""
        return (str, )

    def record(self, data, prefix=''):
        """Log data to text file."""
        out = data
        if self._with_timestamp:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f %Z')
            out = "%s%s%s" % (timestamp, self._delimiter, out)

        self._log_file.write(out + '\n')
        self._log_file.flush()

    def close(self):
        """Close any files used by the output."""
        self._log_file.close()


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
        """Accept TabularInput objects only."""
        return (TabularInput, )

    def record(self, data, prefix=''):
        """Log tabular data to CSV."""
        dictionary = data.primitive_dict

        writer = csv.DictWriter(
            self._log_file, fieldnames=set(dictionary.keys()))

        if not self._tabular_header_written:
            writer.writeheader()
            self._tabular_header_written = True
        writer.writerow(dictionary)
        self._log_file.flush()

    def close(self):
        """Close any files used by the output."""
        self._log_file.close()
