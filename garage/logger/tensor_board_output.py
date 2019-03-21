"""A `garage.logger.LogOutput` for tensorboard.

It receives the input data stream from `garage.logger`, then add them to
tensorboard summary operations through tensorboardX.

Note:
Neither tensorboardX or TensorBoard does not support log parametric
distributions. We add this feature by sampling data from a
`tfp.distributions.Distribution` object.
"""
import functools

import matplotlib.pyplot as plt
import numpy as np
import tensorboardX as tbx
import tensorflow as tf
import tensorflow_probability as tfp

from garage.logger import LogOutput
from garage.logger import TabularInput


class TensorBoardOutput(LogOutput):
    """
    TensorBoard output for logger.

    :param log_dir(str): The save location of the tensorboard event files.
    :param flush_secs(int): How often, in seconds, to flush the added summaries
    and events to disk.
    """

    def __init__(self, log_dir, flush_secs=120, histogram_samples=1e3):
        self._writer = tbx.SummaryWriter(log_dir, flush_secs=flush_secs)
        self._default_step = 0
        self._histogram_samples = int(histogram_samples)
        self._added_graph = False
        self._waiting_for_dump = []

    @property
    def types_accepted(self):
        """Return the types that the logger may pass to this output."""
        return (TabularInput, )

    def record(self, data, prefix=''):
        """
        Add data to tensorboard summary.

        :param data: The data to be logged by the output.
        :param prefix(str): A prefix placed before a log entry in text outputs.
        """
        if isinstance(data, TabularInput):
            self._waiting_for_dump.append(
                functools.partial(self._record_tabular, data, prefix))
        else:
            raise ValueError('Unacceptable type.')

    def _record_tabular(self, data, prefix, step):
        for key, value in data.as_dict.items():
            self._record_kv(prefix + key, value, step)
            data.mark(key)

    def _record_kv(self, key, value, step):
        if isinstance(value, np.ScalarType):
            self._writer.add_scalar(key, value, step)
        elif isinstance(value, plt.Figure):
            self._writer.add_figure(key, value, step)
        elif isinstance(value, tfp.distributions.Distribution):
            samples = tf.get_default_session().run(
                value.sample(self._histogram_samples))
            self._writer.add_histogram(key, samples, step)

    def dump(self, step=None):
        if not self._added_graph:
            # A static graph should only record once.
            self._writer.add_graph(tf.get_default_graph())
            self._added_graph = True

        # Log the tabular inputs, now that we have a step
        for p in self._waiting_for_dump:
            p(step or self._default_step)
        self._waiting_for_dump.clear()

        # Flush output files
        for w in self._writer.all_writers.values():
            w.flush()

        self._default_step += 1

    def close(self):
        """Flush all the events to disk and close the file."""
        self._writer.close()
