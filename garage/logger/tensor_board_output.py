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
import scipy.stats
import tensorboardX as tbx
import tensorflow as tf

from garage.logger import Histogram
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
        return (TabularInput, tf.Graph)

    def record(self, data, prefix=''):
        """
        Add data to tensorboard summary.

        :param data: The data to be logged by the output.
        :param prefix(str): A prefix placed before a log entry in text outputs.
        """
        if isinstance(data, TabularInput):
            self._waiting_for_dump.append(
                functools.partial(self._record_tabular, data))
        elif isinstance(data, tf.Graph):
            self._record_graph(data)
        else:
            raise ValueError('Unacceptable type.')

    def _record_tabular(self, data, step):
        for key, value in data.as_dict.items():
            self._record_kv(key, value, step)
            data.mark(key)

    def _record_kv(self, key, value, step):
        if isinstance(value, np.ScalarType):
            self._writer.add_scalar(key, value, step)
        elif isinstance(value, plt.Figure):
            self._writer.add_figure(key, value, step)
        elif isinstance(value, scipy.stats._distn_infrastructure.rv_frozen):
            shape = (self._histogram_samples, ) + value.mean().shape
            self._writer.add_histogram(key, value.rvs(shape), step)
        elif isinstance(value, scipy.stats._multivariate.multi_rv_frozen):
            self._writer.add_histogram(key, value.rvs(self._histogram_samples),
                                       step)
        elif isinstance(value, Histogram):
            self._writer.add_histogram(key, value, step)

    def _record_graph(self, graph):
        graph_def = graph.as_graph_def(add_shapes=True)
        event = tbx.proto.event_pb2.Event(
            graph_def=graph_def.SerializeToString())
        self._writer.file_writer.add_event(event)

    def dump(self, step=None):
        """Flush summary writer to disk."""
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
