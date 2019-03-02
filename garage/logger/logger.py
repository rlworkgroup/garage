"""This is Garage's logger.

It takes in many different types of input and directs them to the correct
output.

The logger has 4 major steps:

    1. Inputs, such as a simple string or something more complicated like
    TabularInput, are passed to the log() method of an instantiated Logger.

    2. The Logger class checks for any outputs that have been added to it, and
    calls the record() method of any outputs that accept the type of input.

    3. The output (a subclass of LogOutput) receives the input via its record()
    method and handles it in whatever way is expected.

    4. (only in some cases) The dump method is used to dump the output to file.
    Some LogOutput subclasses, like StdOutput, do not instantiate this method,
    but it is necessary for the TensorBoardLogger.


# Here's a demonstration of garage's logger:

from garage.logger import logger

+------+
|logger|
+------+

# Let's add an output to the logger. We want to log to the console, so we'll
#  add a StdOutput.

from garage.logger import StdOutput
logger.add_output(StdOutput())

+------+      +---------+
|logger+------>StdOutput|
+------+      +---------+

# Great! Now we can start logging text.

logger.log("Hello Garage")

# This will go straight to the console as "Hello Garage"

+------+                    +---------+
|logger+---"Hello Garage"--->StdOutput|
+------+                    +---------+

# Let's try adding another output.

from garage.logger import TextOutput
logger.add_output(TextOutput("log_folder/log.txt"))

              +---------+
       +------>StdOutput|
+------+      +---------+
|logger|
+------+      +----------+
       +------>TextOutput|
              +----------+

# And another output.

from garage.logger import CsvOutput
logger.add_output(CsvOutput("log_folder/table.csv"))

              +---------+
       +------>StdOutput|
       |      +---------+
       |
+------+      +----------+
|logger+------>TextOutput|
+------+      +----------+
       |
       |      +---------+
       +------>CsvOutput|
              +---------+

# The logger will record anything passed to logger.log to all outputs that
#  accept its type.

logger.log("test")

                    +---------+
       +---"test"--->StdOutput|
       |            +---------+
       |
+------+            +----------+
|logger+---"test"--->TextOutput|
+------+            +----------+
       |
       |            +---------+
       +-----!!----->CsvOutput|
                    +---------+

# !! Note that the logger knows not to send CsvOutput the string "test"
#  Similarly, more complex objects like tf.tensor won't be sent to (for
#  example) TextOutput.
# This behavior is defined in each output's types_accepted property

# Here's a more complex example.
# TabularInput, instantiated for you as the tabular, can log key/value pairs.

from garage.logger import tabular
tabular.record("key", 72)
tabular.record("foo", "bar")
logger.log(tabular)

                     +---------+
       +---tabular--->StdOutput|
       |             +---------+
       |
+------+             +----------+
|logger+-----!!------>TextOutput|
+------+             +----------+
       |
       |             +---------+
       +---tabular--->CsvOutput|
                     +---------+

# !! Note that the logger knows not to send the tabular to TextOutput
# This behavior is defined in each output's types_accepted property

# Console Output:
---  ---
key  72
foo  bar
---  ---

# Feel free to add your own inputs and outputs to the logger!

"""
from contextlib import contextmanager
from warnings import warn

from gym.utils import colorize

from garage.logger.outputs import LogOutput


class Logger:
    """This is the class that handles logging."""

    def __init__(self):
        self._outputs = []
        self._prefixes = []
        self._prefix_str = ""
        self._warned_once = False

    def log(self, data):
        """Magic method that takes in all different types of input.

        This method is the main API for the logger. Any data to be logged goes
        through this method.

        Any data sent to this method is sent to all outputs that accept its
        type (defined in the types_accepted property).

        :param data: Data to be logged. This can be any type specified in the
         types_accepted property of any of the logger outputs.
        """
        if not self._outputs:
            return self._warn("No outputs have been added to the logger.")

        at_least_one_logged = False
        for output in self._outputs:
            if isinstance(data, output.types_accepted):
                output.record(data, prefix=self._prefix_str)
                at_least_one_logged = True

        if not at_least_one_logged:
            warning = "Log data of type " + type(data).__name__
            warning += " was not accepted by any output"
            return self._warn(warning)

        return None

    def add_output(self, output):
        """Add a new output to the logger.

        All data that is compatible with this output will be sent there.

        :param output: An instantiation of a LogOutput subclass to be added.
        """
        if isinstance(output, type):
            raise ValueError("Output class must be instantiated.")
        elif not isinstance(output, LogOutput):
            raise ValueError("Output object must be a subclass of LogOutput")
        self._outputs.append(output)

    def remove_all(self):
        """Remove all outputs that have been added to this logger."""
        self._outputs.clear()

    def remove_output_type(self, output_type):
        """Remove all outputs of a given type.

        :param output_type: A LogOutput subclass type to be removed.
        """
        self._outputs = [
            output for output in self._outputs
            if not isinstance(output, output_type)
        ]

    def reset_output(self, output):
        """Removes, then re-adds a given output to the logger.

        :param output: An instantiation of a LogOutput subclass to be added.
        """
        self.remove_output_type(type(output))
        self.add_output(output)

    def has_output_type(self, output_type):
        """Check to see if a given logger output is attached to the logger.

        :param output_type: A LogOutput subclass type to be checked for.
        """
        for output in self._outputs:
            if isinstance(output, output_type):
                return True
        return False

    def dump_output_type(self, output_type, step=None):
        """Dump all outputs of the given type.

        :param output_type: A LogOutput subclass type to be dumped.
        :param step: The current run step.
        """
        for output in self._outputs:
            if isinstance(output, output_type):
                output.dump(step=step)

    def dump_all(self, step=None):
        """Dump all outputs connected to the logger.

        :param step: The current run step.
        """
        for output in self._outputs:
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

    def _warn(self, msg):
        """Warns the user using warnings.warn.

        The stacklevel parameter needs to be 3 to ensure the call to logger.log
        is the one printed.
        """
        if not self._warned_once:
            warn(colorize(msg, 'yellow'), self.LogWarning, stacklevel=3)
        self._warned_once = True
        return msg

    def disable_warnings(self):
        """Disable logger warnings for testing."""
        self._warned_once = True

    class LogWarning(UserWarning):
        """Warning class for the Logger."""

        pass
