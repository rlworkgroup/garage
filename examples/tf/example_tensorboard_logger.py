#!/usr/bin/env python3

import tensorflow as tf

from garage.logger import logger, tabular, TensorBoardOutput

logger.add_output(TensorBoardOutput("data/local/histogram_example"))
N = 400
for i in range(N):
    sess = tf.Session()
    sess.__enter__()
    k_val = i / float(N)
    tabular.record("app", k_val)
    logger.log(tabular)
    tabular.clear()
    logger.log(("gass", k_val), record='histogram')
    logger.dump(TensorBoardOutput, step=i)
