#!/usr/bin/env python3

import tensorflow as tf

from garage.logger import HistogramInput, TensorBoardOutput, logger, tabular

tensorboard_output = TensorBoardOutput("data/local/histogram_example")
logger.add_output(tensorboard_output)
N = 400
for i in range(N):
    sess = tf.Session()
    sess.__enter__()
    k_val = i / float(N)
    tabular.record("app", k_val)
    tabular.record("gauss", HistogramInput(k_val))

    logger.log(tabular)
    tabular.clear()

    tensorboard_output.dump(step=i)
