"""This is Garage's logger.

It takes in many different types of input and directs them to the correct
output.
"""
from contextlib import contextmanager
from warnings import warn


class Logger:
    """This is the class that handles logging."""

    def __init__(self):
        self._outputs = []
        self._prefixes = []
        self._prefix_str = ""

    def log(self, data):
        """Magic method that takes in all different types of input.

        Any data sent to this method is sent to all outputs that accept its
        type.

        :param data: Data to be logged. This can be any type specified in the
         types_accepted property of any of the logger outputs.
        """
        if not self._outputs:
            warn("No outputs have been added to the logger.")

        at_least_one_logged = False
        for output in self._outputs:
            if isinstance(data, output.types_accepted):
                output.record(data, prefix=self._prefix_str)
                at_least_one_logged = True

        if not at_least_one_logged:
            warn("Log data of type " + type(data).__name__ +
                 " was not accepted by any output")

    def add_output(self, output):
        """Add a new output to the logger.

        All data that is compatible with this output will be sent there.
        """
        self._outputs.append(output)

    def remove_all(self):
        """Remove all outputs that have been added to this logger."""
        self._outputs.clear()

    def remove_output(self, output_type):
        """Remove all outputs of a given type."""
        self._outputs = [
            output for output in self._outputs
            if not isinstance(output, output_type)
        ]

    def reset_output(self, output):
        """Removes, then re-adds a given output to the logger."""
        self.remove_output(type(output))
        self.add_output(output)

    def has_output(self, output_type):
        """Checks to see if a given logger output is attached to the logger.

        :param output_type: The type of output to check for.
        """
        for output in self._outputs:
            if isinstance(output, output_type):
                return True
        return False

    def dump_output(self, output_type, step=None):
        """Dumps all outputs of the given type.

        :param output_type: The type of output to check for.
        :param step: he
        """
        for output in self._outputs:
            if isinstance(output, output_type):
                output.dump(step=step)

    @contextmanager
    def prefix(self, prefix):
        """Add a prefix to the logger.

        This allows text output to be prepended with a given stack of prefixes.

        Example:
        with logger.prefix('prefix: '):
            logger.log('test_string') # this will have the prefix
        logger.log('test_string2') # this will not have the prefix

        :param prefix: The prefix string to be logged.
        """
        self.push_prefix(prefix)
        try:
            yield
        finally:
            self.pop_prefix()

    def push_prefix(self, prefix):
        """Add prefix to prefix stack.

        :param prefix: The prefix string to be logged.
        """
        self._prefixes.append(prefix)
        self._prefix_str = ''.join(self._prefixes)

    def pop_prefix(self):
        """Pop prefix from prefix stack."""
        del self._prefixes[-1]
        self._prefix_str = ''.join(self._prefixes)
