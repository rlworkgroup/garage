import os

import numpy as np
import tensorflow as tf
from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2

import rllab.misc.logger
from rllab.misc.console import mkdir_p


class TensorBoardOutput:
    def __init__(self):
        self._scalars = tf.Summary()
        self._scope_tensor = {}

        self._default_step = 0
        self._writer = None

    def set_dir(self, dir_name):
        if not dir_name:
            if self._writer:
                self._writer.close()
                self._writer = None
        else:
            mkdir_p(os.path.dirname(dir_name))
            self._writer = tf.summary.FileWriter(dir_name)
            self._default_step = 0
            assert self._writer is not None
            rllab.misc.logger.log("tensorboard data will be logged into:" +
                                  dir_name)

    def dump_tensorboard(self, step=None):
        if not self._writer:
            return
        run_step = self._default_step
        if step:
            run_step = step
        else:
            self._default_step += 1

        self._dump_graph()
        self._dump_scalars(run_step)
        self._dump_tensors()

    def record_scalar(self, key, val):
        self._scalars.value.add(tag=str(key), simple_value=float(val))

    def record_tensor(self, key, val):
        scope = str(key).split('/', 1)[0]
        if scope not in self._scope_tensor:
            self._scope_tensor[scope] = [key]
        else:
            if key not in self._scope_tensor[scope]:
                self._scope_tensor[scope].append(key)

        for idx, v in np.ndenumerate(np.array(val)):
            self._scalars.value.add(
                tag=key + '/' + str(idx).strip('()'), simple_value=float(v))

    def _dump_graph(self):
        self._writer.add_graph(tf.get_default_graph())
        self._writer.flush()

    def _dump_scalars(self, step):
        self._writer.add_summary(self._scalars, int(step))
        self._writer.flush()
        del self._scalars.value[:]

    def _dump_tensors(self):
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
            layout_summary = summary_lib.custom_scalar_pb(
                layout_pb2.Layout(category=layout_categories))
            self._writer.add_summary(layout_summary)
