"""Contains the tabular class.

This class holds tabular information for later output.
"""
from contextlib import contextmanager
import numbers

import numpy as np
import tabulate


class TabularInput:
    """This class allows the user to create tables for easy display.

    TabularInput may be passed to the logger via its log() method.
    """

    def __init__(self):
        self._dict = {}
        self._prefixes = []
        self._prefix_str = ''

    def __str__(self):
        """Return a string representation of the table for the logger."""
        return tabulate.tabulate(self.primitive_dict.items())

    def record(self, key, val):
        """Save key/value entries for the table.

        :param key: String key corresponding to the value.
        :param val: Value that is to be stored in the table.
        """
        self._dict[self._prefix_str + str(key)] = val

    def record_misc_stat(self, key, values, placement='back'):
        """Record statistics of an array.

        :param key: String key corresponding to the values.
        :param values: Array of values to be analyzed.
        :param placement: Whether to put the prefix in front or in the back.
        """
        if placement == 'front':
            front = ""
            back = key
        else:
            front = key
            back = ""
        if values:
            self.record(front + "Average" + back, np.average(values))
            self.record(front + "Std" + back, np.std(values))
            self.record(front + "Median" + back, np.median(values))
            self.record(front + "Min" + back, np.min(values))
            self.record(front + "Max" + back, np.max(values))
        else:
            self.record(front + "Average" + back, np.nan)
            self.record(front + "Std" + back, np.nan)
            self.record(front + "Median" + back, np.nan)
            self.record(front + "Min" + back, np.nan)
            self.record(front + "Max" + back, np.nan)

    @contextmanager
    def prefix(self, prefix):
        """Handle pushing and popping of a tabular prefix.

        Can be used in the following way:

        with tabular.prefix('your_prefix_'):
            # your code
            tabular.record(key, val)

        :param prefix: The string prefix to be prepended to logs.
        """
        self.push_prefix(prefix)
        try:
            yield
        finally:
            self.pop_prefix()

    def clear(self):
        """Clear the tabular."""
        self._dict.clear()

    def push_prefix(self, prefix):
        """Push prefix to be appended before printed table.

        :param prefix: The string prefix to be prepended to logs.
        """
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def pop_prefix(self):
        """Pop prefix that was appended to the printed table."""
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)

    @property
    def primitive_dict(self):
        """Return the dictionary, excluding all nonprimitive types."""
        primitives = (int, float, str, bool)

        def is_primitive(x):
            if isinstance(x, primitives) or isinstance(x, numbers.Number):
                return True
            elif type(x).__module__ == np.__name__:
                try:
                    return np.issubdtype(x.dtype, np.generic)
                except AttributeError:
                    return False
            else:
                return False

        return {
            key: val
            for key, val in self._dict.items() if is_primitive(val)
        }

    @property
    def dict(self):
        """Return a dictionary of the tabular items."""
        return self._dict

    @property
    def key_set(self):
        """Return a set of the table's keys."""
        return set(self._dict.keys())
