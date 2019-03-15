"""A `garage.logger.LogOutput` for CSV files."""
import csv
import warnings

from garage.logger import TabularInput
from garage.logger.outputs import FileOutput
from garage.misc.console import colorize


class CsvOutput(FileOutput):
    """CSV file output for logger.

    :param file_name: The file this output should log to.
    """

    def __init__(self, file_name):
        super().__init__(file_name)
        self._warned_once = False
        self._writer = None
        self._fieldnames = None

    @property
    def types_accepted(self):
        """Accept TabularInput objects only."""
        return (TabularInput, )

    def record(self, data, prefix=''):
        """Log tabular data to CSV."""
        to_csv = data.as_primitive_dict

        if not self._writer:
            self._fieldnames = set(to_csv.keys())
            self._writer = csv.DictWriter(
                self._log_file,
                fieldnames=self._fieldnames,
                extrasaction='ignore')
            self._writer.writeheader()

        if to_csv.keys() != self._fieldnames:
            self._warn('Inconsistent TabularInput keys detected. '
                       'CsvOutput keys: {}. '
                       'TabularInput keys: {}. '
                       'Did you change key sets after your first '
                       'logger.log(TabularInput)?'.format(
                           set(self._fieldnames), set(to_csv.keys())))

        self._writer.writerow(to_csv)
        self._log_file.flush()

    def _warn(self, msg):
        """Warns the user using warnings.warn.

        The stacklevel parameter needs to be 3 to ensure the call to logger.log
        is the one printed.
        """
        if not self._warned_once:
            warnings.warn(
                colorize(msg, 'yellow'), CsvOutputWarning, stacklevel=3)
        self._warned_once = True
        return msg

    def disable_warnings(self):
        """Disable logger warnings for testing."""
        self._warned_once = True


class CsvOutputWarning(UserWarning):
    """Warning class for the Logger."""

    pass
