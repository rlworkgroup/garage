"""A `garage.logger` input for tabular (key-value) data."""
import contextlib
import warnings

import numpy as np
import tabulate

from garage.misc.console import colorize


class TabularInput:
    """This class allows the user to create tables for easy display.

    TabularInput may be passed to the logger via its log() method.
    """

    def __init__(self):
        self._dict = {}
        self._recorded = set()
        self._prefixes = []
        self._prefix_str = ''
        self._warned_once = set()
        self._disable_warnings = False

    def __str__(self):
        """Return a string representation of the table for the logger."""
        return tabulate.tabulate(
            sorted(self.as_primitive_dict.items(), key=lambda x: x[0]))

    def record(self, key, val):
        """Save key/value entries for the table.

        :param key: String key corresponding to the value.
        :param val: Value that is to be stored in the table.
        """
        self._dict[self._prefix_str + str(key)] = val

    def mark(self, key):
        self._recorded.add(key)

    def mark_str(self):
        self._recorded |= self.as_primitive_dict.keys()

    def mark_all(self):
        self._recorded |= self._dict.keys()

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
            self.record(front + 'Average' + back, np.average(values))
            self.record(front + 'Std' + back, np.std(values))
            self.record(front + 'Median' + back, np.median(values))
            self.record(front + 'Min' + back, np.min(values))
            self.record(front + 'Max' + back, np.max(values))
        else:
            self.record(front + 'Average' + back, np.nan)
            self.record(front + 'Std' + back, np.nan)
            self.record(front + 'Median' + back, np.nan)
            self.record(front + 'Min' + back, np.nan)
            self.record(front + 'Max' + back, np.nan)

    @contextlib.contextmanager
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

        # Warn if something wasn't logged
        for k, v in self._dict.items():
            if k not in self._recorded:
                warning = (
                    'TabularInput {{{}: type({})}} was not accepted by any output'
                    .format(k,
                            type(v).__name__))
                self._warn(warning)

        self._dict.clear()
        self._recorded.clear()

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
    def as_primitive_dict(self):
        """Return the dictionary, excluding all nonprimitive types."""
        return {
            key: val
            for key, val in self._dict.items() if np.isscalar(val)
        }

    @property
    def as_dict(self):
        """Return a dictionary of the tabular items."""
        return self._dict

    def _warn(self, msg):
        """Warns the user using warnings.warn.

        The stacklevel parameter needs to be 3 to ensure the call to logger.log
        is the one printed.
        """
        if not self._disable_warnings and msg not in self._warned_once:
            warnings.warn(
                colorize(msg, 'yellow'), TabularInputWarning, stacklevel=3)
        self._warned_once.add(msg)
        return msg

    def disable_warnings(self):
        """Disable logger warnings for testing."""
        self._disable_warnings = True


class TabularInputWarning(UserWarning):
    """Warning class for the TabularInpuit"""

    pass
