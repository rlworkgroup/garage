"""Contains the tabular class.

This class allows the recording of tabular information for later output.
TabularInput may be passed to the logger via its log() method.
"""
from contextlib import contextmanager

import numpy as np
import tabulate


class TabularInput(object):
    """This class allows the user to create tables for easy display."""

    def __init__(self):
        self._tabular = []
        self._no_prefix_dict = {}
        self._prefixes = []
        self._prefix_str = ''

    def __str__(self):
        """Returns a string representation of the table for the logger."""
        return tabulate.tabulate(self._tabular)

    def record(self, key, val):
        """Allows the user to save key/value entries for the table."""
        self._tabular.append((self._prefix_str + str(key), str(val)))
        self._no_prefix_dict[key] = val

    def record_misc_stat(self, key, values, placement='back'):
        """Allows the user to record statistics of an array."""
        if placement == 'front':
            prefix = ""
            suffix = key
        else:
            prefix = key
            suffix = ""
        if values:
            self.record(prefix + "Average" + suffix, np.average(values))
            self.record(prefix + "Std" + suffix, np.std(values))
            self.record(prefix + "Median" + suffix, np.median(values))
            self.record(prefix + "Min" + suffix, np.min(values))
            self.record(prefix + "Max" + suffix, np.max(values))
        else:
            self.record(prefix + "Average" + suffix, np.nan)
            self.record(prefix + "Std" + suffix, np.nan)
            self.record(prefix + "Median" + suffix, np.nan)
            self.record(prefix + "Min" + suffix, np.nan)
            self.record(prefix + "Max" + suffix, np.nan)

    @contextmanager
    def prefix(self, key):
        """Handles pushing and popping of a tabular prefix.

        Can be used in the following way:

        with tabular.prefix('your_prefix_'):
            # your code
            tabular.record_tabular(key, val)
        """
        self.push_prefix(key)
        try:
            yield
        finally:
            self.pop_prefix()

    def clear(self):
        """Clears the tabular."""
        self._tabular.clear()

    def push_prefix(self, key):
        """Push prefix to be appended before printed table."""
        self._prefixes.append(key)
        self._prefix_str = ''.join(self._prefixes)

    def pop_prefix(self, ):
        """Pop prefix that was appended to the printed table."""
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)

    def get_table_dict(self):
        """Returns a dictionary of the tabular items."""
        return dict(self._tabular)

    def get_no_prefix_dict(self):
        """Returns dictionary without prefixes."""
        return self._no_prefix_dict

    def get_table_key_set(self):
        """Returns a set of the table's keys."""
        return set(dict(self._tabular).keys())
