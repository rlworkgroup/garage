from garage.misc.logger.base_logger import Logger
from garage.misc.logger.tensorboard_output import TensorBoardOutput


class TensorboardLogger(Logger):
    def __init__(self):
        super(TensorboardLogger, self).__init__()
        self._tensorboard_step_key = None
        self._tensorboard = TensorBoardOutput()

    def set_tensorboard_dir(self, dir_name):
        self._tensorboard.set_dir(dir_name)
        self.log("tensorboard data will be logged into:" + dir_name)

    def set_tensorboard_step_key(self, key):
        self._tensorboard_step_key = key

    def record_tabular(self, key, val):
        self._tensorboard.record_scalar(str(key), val)
        self._tabular.append((self._tabular_prefix_str + str(key), str(val)))

    def record_tensor(self, key, val):
        """Record tf.Tensor into tensorboard with Tensor.name and its value."""
        self._tensorboard.record_tensor(key, val)

    def record_histogram(self, key, val):
        self._tensorboard.record_histogram(str(key), val)

    def record_histogram_by_type(self,
                                 histogram_type,
                                 key=None,
                                 shape=[1000],
                                 **kwargs):
        self._tensorboard.record_histogram_by_type(histogram_type, key, shape,
                                                   **kwargs)

    def dump_tensorboard(self, *args, **kwargs):
        if self._tabular:
            tabular_dict = dict(self._tabular)
        step = None
        if self._tensorboard_step_key and self._tensorboard_step_key in tabular_dict:
            step = tabular_dict[self._tensorboard_step_key]
        self._tensorboard.dump_tensorboard(step)

    def dump_tabular(self, *args, **kwargs):
        super(TensorboardLogger, self).dump_tabular(*args, **kwargs)

        # write to the tensorboard folder
        # This assumes that the keys in each iteration won't change!
        self.dump_tensorboard(args, kwargs)

    def reset(self):
        super(TensorboardLogger, self).reset()
        self._tensorboard = TensorBoardOutput()
