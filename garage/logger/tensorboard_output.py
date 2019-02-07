"""Contains the output class for tensorboard.

This class is sent logger data and handles the transfer to tensorboard.

Note:
Most of the methods were brought in from the old logger and should probably
be rewritten with tensorboardX
"""
from os.path import abspath, dirname
import shutil

import google.protobuf.json_format as json_format
from jsonmerge import merge
import numpy as np
from tensorboard import summary as summary_lib
from tensorboard.backend.event_processing import (plugin_event_multiplexer as
                                                  event_multiplexer)
from tensorboard.plugins.custom_scalar import layout_pb2, metadata
import tensorflow as tf

from garage import config
from garage.logger import (HistogramInput, HistogramInputDistribution,
                           LogOutput, TabularInput)
from garage.misc.console import mkdir_p


class TensorBoardOutput(LogOutput):
    """Tensorboard output class for the logger.

    :param log_dir: The directory this output should log to.
    """

    def __init__(self, log_dir=config.LOG_DIR):
        self._scalars = tf.Summary()
        self._scope_tensor = {}
        self._has_recorded_tensor = False
        self._has_dumped_graph = False

        self._histogram_ds = {}
        self._histogram_summary_op = []
        self._histogram_distribute_list = [
            'normal', 'gamma', 'poisson', 'uniform'
        ]
        self._feed = {}

        self._default_step = 0
        self._writer = None
        self._writer_dir = None
        self._layout_writer = None
        self._layout_writer_dir = None

        self._set_dir(log_dir)

    @property
    def types_accepted(self):
        """The types that the logger may pass to this output."""
        return (TabularInput, )

    def record(self, data, prefix=''):
        if isinstance(data, TabularInput):
            for key, value in data.dict.items():
                if isinstance(value, HistogramInput):
                    self.record_histogram(key, value.data)
                elif isinstance(value, HistogramInputDistribution):
                    self.record_histogram_by_type(**vars(value))
                elif isinstance(value, tf.Tensor):
                    self.record_tensor(key, value)
                else:
                    self.record_scalar(key, value)

    def _set_dir(self, dir_name):
        if not dir_name:
            if self._writer:
                self._writer.close()
                self._writer = None
        else:
            mkdir_p(dirname(dir_name))
            self._writer_dir = dir_name
            self._writer = tf.summary.FileWriter(dir_name)

            self._layout_writer_dir = dirname(dirname(
                abspath(dir_name))) + '/custom_scalar_config'

            self._default_step = 0
            assert self._writer is not None

    def dump(self, step=None):
        """Dumps this output to the log dir.

        :param step: The current run step.
        """
        if not self._writer:
            return
        run_step = self._default_step
        if step:
            run_step = step
        else:
            self._default_step += 1

        self._dump_graph()
        self._dump_scalars(run_step)
        self._dump_histogram(run_step)
        self._dump_tensors()

    def record_histogram(self, key, val, name=None):
        if str(key) not in self._histogram_ds:
            with tf.name_scope(name, "record_histogram"):
                self._histogram_ds[str(key)] = tf.Variable(val)
            self._histogram_summary_op.append(
                tf.summary.histogram(str(key), self._histogram_ds[str(key)]))
            self._histogram_summary_op_merge = tf.summary.merge(
                self._histogram_summary_op)

        self._feed[self._histogram_ds[str(key)]] = val

    def record_histogram_by_type(self,
                                 histogram_type,
                                 key=None,
                                 shape=[1000],
                                 name=None,
                                 **kwargs):
        '''
        distribution type and args:
            normal: mean, stddev
            gamma: alpha
            poisson: lam
            uniform: maxval

        example:
            $ python examples/example_tensorboard_logger.py
        '''
        if histogram_type not in self._histogram_distribute_list:
            raise Exception('histogram type error %s' % histogram_type,
                            'builtin type', self._histogram_distribute_list)

        if str(key) not in self._histogram_ds:
            self._histogram_ds[str(key)] = self._get_histogram_var_by_type(
                histogram_type, shape, name, **kwargs)
            self._histogram_summary_op.append(
                tf.summary.histogram(
                    str(key), self._histogram_ds[str(key)][0]))
            self._histogram_summary_op_merge = tf.summary.merge(
                self._histogram_summary_op)

        key_list = self._histogram_ds[str(key)][1]
        val_list = self._get_histogram_val_by_type(histogram_type, **kwargs)

        for key, val in zip(key_list, val_list):
            self._feed[key] = val

    def record_scalar(self, key, val):
        self._scalars.value.add(tag=str(key), simple_value=float(val))

    def record_tensor(self, key, val):
        self._has_recorded_tensor = True
        scope = str(key).split('/', 1)[0]
        if scope not in self._scope_tensor:
            self._scope_tensor[scope] = [key]
        else:
            if key not in self._scope_tensor[scope]:
                self._scope_tensor[scope].append(key)

        for idx, v in np.ndenumerate(np.array(val)):
            self._scalars.value.add(
                tag=key + '/' + str(idx).strip('()'), simple_value=float(v))

    def _get_histogram_var_by_type(self,
                                   histogram_type,
                                   shape,
                                   name=None,
                                   **kwargs):
        with tf.name_scope(name, "get_hist_{}".format(histogram_type)):
            if histogram_type == "normal":
                # Make a normal distribution, with a shifting mean
                mean = tf.Variable(kwargs['mean'])
                stddev = tf.Variable(kwargs['stddev'])
                return tf.random_normal(
                    shape=shape, mean=mean, stddev=stddev), [mean, stddev]
            elif histogram_type == "gamma":
                # Add a gamma distribution
                alpha = tf.Variable(kwargs['alpha'])
                return tf.random_gamma(shape=shape, alpha=alpha), [alpha]
            elif histogram_type == "poisson":
                lam = tf.Variable(kwargs['lam'])
                return tf.random_poisson(shape=shape, lam=lam), [lam]
            elif histogram_type == "uniform":
                # Add a uniform distribution
                maxval = tf.Variable(kwargs['maxval'])
                return tf.random_uniform(shape=shape, maxval=maxval), [maxval]

            raise Exception('histogram type error %s' % histogram_type,
                            'builtin type', self._histogram_distribute_list)

    def _get_histogram_val_by_type(self, histogram_type, **kwargs):
        if histogram_type == "normal":
            # Make a normal distribution, with a shifting mean
            return [kwargs['mean'], kwargs['stddev']]
        elif histogram_type == "gamma":
            # Add a gamma distribution
            self.alpha_v = kwargs['alpha']
            return [kwargs['alpha']]
        elif histogram_type == "poisson":
            return [kwargs['lam']]
        elif histogram_type == "uniform":
            # Add a uniform distribution
            return [kwargs['maxval']]

        raise Exception('histogram type error %s' % histogram_type,
                        'builtin type', self._histogram_distribute_list)

    def _dump_graph(self):
        # We only need to write the graph event once (instead of per step).
        if self._has_dumped_graph:
            return

        self._has_dumped_graph = True
        self._writer.add_graph(tf.get_default_graph())
        self._writer.flush()

    def _dump_scalars(self, step):
        self._writer.add_summary(self._scalars, int(step))
        self._writer.flush()
        del self._scalars.value[:]

    def _dump_histogram(self, step):
        self.session = tf.get_default_session()
        if self._histogram_summary_op:
            summary_str = self.session.run(
                self._histogram_summary_op_merge, feed_dict=self._feed)
            self._writer.add_summary(summary_str, global_step=step)
            self._writer.flush()

    def _dump_tensors(self):
        if not self._has_recorded_tensor:
            return

        layout_categories = []

        for scope in self._scope_tensor:
            chart = []
            for name in self._scope_tensor[scope]:
                chart.append(
                    layout_pb2.Chart(
                        title=name,
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'name(?!.*margin.*)'.replace('name', name)
                                 ])))
            category = layout_pb2.Category(title=scope, chart=chart)
            layout_categories.append(category)

        if layout_categories:
            layout_proto_to_write = layout_pb2.Layout(
                category=layout_categories)

            try:
                # Load former layout_proto from self._layout_writer_dir.
                multiplexer = event_multiplexer.EventMultiplexer()
                multiplexer.AddRunsFromDirectory(self._layout_writer_dir)
                multiplexer.Reload()
                tensor_events = multiplexer.Tensors(
                    '.', metadata.CONFIG_SUMMARY_TAG)
                shutil.rmtree(self._layout_writer_dir)

                # Parse layout proto from disk.
                string_array = tf.make_ndarray(tensor_events[0].tensor_proto)
                content = np.asscalar(string_array)
                layout_proto_from_disk = layout_pb2.Layout()
                layout_proto_from_disk.ParseFromString(
                    tf.compat.as_bytes(content))

                # Merge two layout proto.
                merged_layout_json = merge(
                    json_format.MessageToJson(layout_proto_from_disk),
                    json_format.MessageToJson(layout_proto_to_write))
                merged_layout_proto = layout_pb2.Layout()
                json_format.Parse(str(merged_layout_json), merged_layout_proto)

                self._layout_writer = tf.summary.FileWriter(
                    self._layout_writer_dir)
                layout_summary = summary_lib.custom_scalar_pb(
                    merged_layout_proto)
                self._layout_writer.add_summary(layout_summary)
                self._layout_writer.close()
            except KeyError:
                # Write the current layout proto into disk
                # when there is no layout.
                self._layout_writer = tf.summary.FileWriter(
                    self._layout_writer_dir)
                layout_summary = summary_lib.custom_scalar_pb(
                    layout_proto_to_write)
                self._layout_writer.add_summary(layout_summary)
                self._layout_writer.close()
