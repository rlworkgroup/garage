"""A `garage.logger.LogOutput` for tensorboard.

It receives the input data stream from `garage.logger`, then add them to
tensorboard summary operations through tensorboardX.

Note:
Neither tensorboardX or TensorBoard does not support log parametric
distributions. We add this feature by sampling data from a
`tfp.distributions.Distribution` object.
"""
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import tensorflow as tf
import tensorflow_probability as tfp

from garage import config
from garage.logger import LogOutput
from garage.logger import TabularInput


class TensorBoardOutput(LogOutput):
    """
    TensorBoard output for logger.

    :param log_dir(str): The save location of the tensorboard event files.
    :param flush_secs(int): How often, in seconds, to flush the added summaries
    and events to disk.
    """

    def __init__(self,
                 log_dir=config.GARAGE_LOG_DIR,
                 flush_secs=120,
                 histogram_samples=int(1e3)):
        self._writer = SummaryWriter(log_dir, flush_secs=flush_secs)
        self._default_step = 0
        self._histogram_samples = histogram_samples
        self._added_graph = False

    @property
    def types_accepted(self):
        """Return the types that the logger may pass to this output."""
        return (TabularInput, tfp.distributions.Distribution)

    def record(self, data, prefix='', step=None):
        """
        Add data to tensorboard summary.

        :param data(garage.logger.input): The data to be logged by the output.
        :param prefix(str): A prefix placed before a log entry in text outputs.
        """
        if not step:
            step = self._default_step
            self._default_step += 1

        if not self._added_graph:
            # A static graph should only record once.
            self._writer.add_graph(tf.get_default_graph())
            self._added_graph = False

        if isinstance(data, TabularInput):
            for key, value in data.as_dict.items():
                if np.isscalar(value):
                    self._writer.add_scalar(key, value, step)
                elif isinstance(value, plt.Figure):
                    self._writer.add_figure(key, value, step)
                else:
                    raise ValueError('Unknown value type: {}'.format(
                        type(value).__name__))
        elif isinstance(data, tfp.distributions.Distribution):
            values = tf.get_default_session().run(
                data.sample(self._histogram_samples))
            self._writer.add_histogram(data.name, values, step)
        else:
            raise ValueError('Unknown data input: {}. '
                             'TensorBoardOutput supports: {}'.format(
                                 type(data).__name__, self.types_accepted))

    def close(self):
        """Flush all the events to disk and close the file."""
        self._writer.close()
