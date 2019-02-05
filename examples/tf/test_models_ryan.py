#!/usr/bin/env python3
import pickle

import ipdb
import numpy as np
import tensorflow as tf

from garage.tf.models.gaussian_mlp_model_ryan import GaussianMLPModel


# Construction can happen outside a session
g = GaussianMLPModel("model1")

# Call has to happen inside a session
sess = tf.Session()
with sess.as_default():
    state = tf.placeholder(shape=[None, 10, 5], dtype=tf.float32)
    outs = g.build(state)

    # sess.run(tf.global_variables_initializer())

    print('name: {}'.format(g.name))
    print('weights: {}'.format(g.weights))

    print('zeroing weights...')
    g.weights = {k: np.zeros_like(v) for k, v in g.weights.items()}
    print('weights (after zeroing): {}'.format(g.weights))

    # sess.run(tf.global_variables_initializer())
    print('weights (after reset): {}'.format(g.weights))

    print('settings weights to 1...')
    g.weights = {k: np.ones_like(v) for k, v in g.weights.items()}
    print('pickle data: {}'.format(g.__getstate__()))
    h_data = pickle.dumps(g)
sess.close()

tf.reset_default_graph()

sess = tf.Session()
with sess.as_default():
    h = pickle.loads(h_data)
    state2 = tf.placeholder(shape=[None, 10, 5], dtype=tf.float32)
    outs = h.build(state2)

    # sess.run(tf.global_variables_initializer())

    print('weights (after pickling): {}'.format(h.weights))
    ipdb.set_trace()
    print('done!')
