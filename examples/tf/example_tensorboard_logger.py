#!/usr/bin/env python3

import tensorflow as tf

from garage.logger import distributions, logger, tabular, TensorBoardOutput

tensor_board_output = TensorBoardOutput("data/local/histogram_example")
logger.add_output(tensor_board_output)
N = 400
for i in range(N):
    sess = tf.Session()
    sess.__enter__()
    k_val = i / float(N)
    tabular.record("app", k_val)
    tabular.record("gauss", distributions.Empirical(k_val))

    logger.log(tabular)
    tabular.clear()

    tensor_board_output.dump(step=i)
