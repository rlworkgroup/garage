#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from garage.misc import logger

logger.set_tensorboard_dir("data/local/histogram_example")
N = 400
for i in range(N):
    sess = tf.Session()
    sess.__enter__()
    logger.log("Starting epoch {}".format(i))
    k_val = i / float(N)
    logger.record_histogram_by_type('gamma', key='gamma', alpha=k_val)
    logger.record_histogram_by_type(
        'normal', key='normal', mean=5 * k_val, stddev=1.0)
    logger.record_histogram_by_type('poisson', key='poisson', lam=k_val)
    logger.record_histogram_by_type(
        'uniform', key='uniform', maxval=k_val * 10)
    logger.record_tabular("app", k_val)
    logger.record_histogram("gass", k_val)

    # Make a random plot
    fig = plt.figure()
    ax = fig.gca()
    xs = np.arange(10.0)
    ys = np.random.rand(10)
    ax.scatter(xs, ys)
    logger.record_matplotlib('foo', fig)
    plt.close(fig)

    logger.dump_tensorboard(step=i)
