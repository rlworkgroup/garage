import os

import tensorflow as tf

from rllab.misc.console import mkdir_p


class Summary:
    def __init__(self):
        self._summary_scale = tf.Summary()
        self._histogram_ds = {}
        self._histogram_summary_op = []

        self._session = tf.Session()

        self._default_step = 0
        self._step_key = None
        self._writer = None
        self._histogram_distribute_list = [
            'normal', 'gamma', 'poisson', 'uniform'
        ]
        self._feed = {}

    def record_histogram(self, key, val):
        if str(key) not in self._histogram_ds:
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
                                 **kwargs):
        '''
        distribution type and args:
            normal: mean, stddev
            gamma: alpha
            poisson: lam
            uniform: maxval
        '''
        if histogram_type not in self._histogram_distribute_list:
            raise Exception('histogram type error %s' % histogram_type,
                            'builtin type', self._histogram_distribute_list)

        if str(key) not in self._histogram_ds:
            self._histogram_ds[str(key)] = self._get_histogram_var_by_type(
                histogram_type, shape, **kwargs)
            self._histogram_summary_op.append(
                tf.summary.histogram(
                    str(key), self._histogram_ds[str(key)][0]))
            self._histogram_summary_op_merge = tf.summary.merge(
                self._histogram_summary_op)

        key_list = self._histogram_ds[str(key)][1]
        val_list = self._get_histogram_val_by_type(histogram_type, **kwargs)

        for key, val in zip(key_list, val_list):
            self._feed[key] = val

    def record_scale(self, key, val):
        self._summary_scale.value.add(tag=str(key), simple_value=float(val))

    def dump_tensorboard(self, step=None):
        if not self._writer:
            return
        run_step = self._default_step
        if step:
            run_step = step
        else:
            self._default_step += 1

        self._dump_histogram(run_step)
        self._dump_scale(run_step)

    def set_dir(self, dir_name):
        if not dir_name:
            if self._writer:
                self._writer.close()
                self._writer = None
        else:
            if dir_name[-1] != '/':
                dir_name += '/'
            mkdir_p(os.path.dirname(dir_name))
            self._writer = tf.summary.FileWriter(dir_name)
            self._default_step = 0
            assert self._writer is not None
            print("tensorboard data will be logged into:", dir_name)

    def _dump_histogram(self, step):
        if len(self._histogram_summary_op):
            summary_str = self._session.run(
                self._histogram_summary_op_merge, feed_dict=self._feed)
            self._writer.add_summary(summary_str, global_step=step)
            self._writer.flush()

    def _dump_scale(self, step):
        self._writer.add_summary(self._summary_scale, step)
        self._writer.flush()
        del self._summary_scale.value[:]

    def _get_histogram_var_by_type(self, histogram_type, shape, **kwargs):
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
            # And a uniform distribution
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
            # And a uniform distribution
            return [kwargs['maxval']]

        raise Exception('histogram type error %s' % histogram_type,
                        'builtin type', self._histogram_distribute_list)
