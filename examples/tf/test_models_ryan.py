import pickle

import numpy as np
import tensorflow as tf

from garage.tf.models.gaussian_mlp_model_ryan import GaussianMLPModel

# Construction can happen outside a session
g = GaussianMLPModel(output_dim=2)

# Call has to happen inside a session
sess = tf.Session()
with sess.as_default():
    state = tf.placeholder(shape=[None, 10, 5], dtype=tf.float32)
    outs = g.build(state)

    # sess.run(tf.global_variables_initializer())

    print('name: {}'.format(g.name))
    print('parameters: {}'.format(g.parameters))

    print('zeroing parameters...')
    g.parameters = {k: np.zeros_like(v) for k, v in g.parameters.items()}
    print('parameters (after zeroing): {}'.format(g.parameters))

    print('settings parameters to 1...')
    g.parameters = {k: np.ones_like(v) for k, v in g.parameters.items()}
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

    print('parameters (after pickling): {}'.format(h.parameters))
    print('done!')
