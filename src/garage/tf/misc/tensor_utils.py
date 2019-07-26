from collections import Iterable
from collections import namedtuple

import numpy as np
import tensorflow as tf


def compile_function(inputs, outputs, log_name=None):
    def run(*input_vals):
        sess = tf.compat.v1.get_default_session()
        return sess.run(outputs, feed_dict=dict(list(zip(inputs, input_vals))))

    return run


def get_target_ops(variables, target_variables, tau=None):
    """
    Get target variables update operations.

    In RL algorithms we often update target network every n
    steps. This function returns the tf.Operation for updating
    target variables (denoted by target_var) from variables
    (denote by var) with fraction tau. In other words, each time
    we want to keep tau of the var and add (1 - tau) of target_var
    to var.

    Args:
        variables (list[tf.Variable]): Soure variables for update.
        target_variable (list[tf.Variable]): Target variables to
            be updated.
        tau (float): Fraction to update. Set it to be None for
            hard-update.
    """
    update_ops = []
    init_ops = []
    assert len(variables) == len(target_variables)
    for var, target_var in zip(variables, target_variables):
        init_ops.append(tf.compat.v1.assign(target_var, var))
        if tau is not None:
            update_ops.append(
                tf.compat.v1.assign(target_var,
                                    tau * var + (1.0 - tau) * target_var))

    if tau is not None:
        return init_ops, update_ops
    else:
        return init_ops


def flatten_batch(t, name='flatten_batch'):
    return tf.reshape(t, [-1] + list(t.shape[2:]), name=name)


def flatten_batch_dict(d, name=None):
    with tf.name_scope(name, 'flatten_batch_dict', [d]):
        return {k: flatten_batch(v) for k, v in d.items()}


def filter_valids(t, valid, name='filter_valids'):
    # 'valid' is either 0 or 1 with dtype of tf.float32
    # Must round before cast to prevent floating-error
    return tf.dynamic_partition(
        t, tf.cast(tf.round(valid), tf.int32), 2, name=name)[1]


def filter_valids_dict(d, valid, name=None):
    with tf.name_scope(name, 'filter_valids_dict', [d, valid]):
        return {k: filter_valids(v, valid) for k, v in d.items()}


def graph_inputs(name, **kwargs):
    Singleton = namedtuple(name, kwargs.keys())
    return Singleton(**kwargs)


def flatten_inputs(deep):
    def flatten(deep):
        for d in deep:
            if isinstance(d, Iterable) and not isinstance(
                    d, (str, bytes, tf.Tensor, np.ndarray)):
                yield from flatten(d)
            else:
                yield d

    return list(flatten(deep))


def flatten_tensor_variables(ts):
    return tf.concat(
        axis=0,
        values=[tf.reshape(x, [-1]) for x in ts],
        name='flatten_tensor_variables')


def unflatten_tensor_variables(flatarr, shapes, symb_arrs):
    arrs = []
    n = 0
    for (shape, symb_arr) in zip(shapes, symb_arrs):
        size = np.prod(list(shape))
        arr = tf.reshape(flatarr[n:n + size], shape)
        arrs.append(arr)
        n += size
    return arrs


def new_tensor(name, ndim, dtype):
    return tf.compat.v1.placeholder(
        dtype=dtype, shape=[None] * ndim, name=name)


def new_tensor_like(name, arr_like):
    return new_tensor(name,
                      arr_like.get_shape().ndims, arr_like.dtype.base_dtype)


def concat_tensor_list(tensor_list):
    return np.concatenate(tensor_list, axis=0)


def concat_tensor_dict_list(tensor_dict_list):
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = concat_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def stack_tensor_list(tensor_list):
    return np.array(tensor_list)
    # tensor_shape = np.array(tensor_list[0]).shape
    # if tensor_shape is tuple():
    #     return np.array(tensor_list)
    # return np.vstack(tensor_list)


def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary
     of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def split_tensor_dict_list(tensor_dict):
    keys = list(tensor_dict.keys())
    ret = None
    for k in keys:
        vals = tensor_dict[k]
        if isinstance(vals, dict):
            vals = split_tensor_dict_list(vals)
        if ret is None:
            ret = [{k: v} for v in vals]
        else:
            for v, cur_dict in zip(vals, ret):
                cur_dict[k] = v
    return ret


def pad_tensor(x, max_len):
    return np.concatenate([
        x,
        np.tile(
            np.zeros_like(x[0]), (max_len - len(x), ) + (1, ) * np.ndim(x[0]))
    ])


def pad_tensor_n(xs, max_len):
    ret = np.zeros((len(xs), max_len) + xs[0].shape[1:], dtype=xs[0].dtype)
    for idx, x in enumerate(xs):
        ret[idx][:len(x)] = x
    return ret


def pad_tensor_dict(tensor_dict, max_len):
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = pad_tensor_dict(tensor_dict[k], max_len)
        else:
            ret[k] = pad_tensor(tensor_dict[k], max_len)
    return ret


def compute_advantages(discount,
                       gae_lambda,
                       max_len,
                       baselines,
                       rewards,
                       name=None):
    with tf.name_scope(name, 'compute_advantages',
                       [discount, gae_lambda, max_len, baselines, rewards]):
        # Calculate advantages
        #
        # Advantages are a discounted cumulative sum.
        #
        # The discount cumulative sum can be represented as an IIR
        # filter ob the reversed input vectors, i.e.
        #    y[t] - discount*y[t+1] = x[t]
        #        or
        #    rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
        #
        # Given the time-domain IIR filter step response, we can
        # calculate the filter response to our signal by convolving the
        # signal with the filter response function. The time-domain IIR
        # step response is calculated below as discount_filter:
        #     discount_filter =
        #         [1, discount, discount^2, ..., discount^N-1]
        #         where the epsiode length is N.
        #
        # We convolve discount_filter with the reversed time-domain
        # signal deltas to calculate the reversed advantages:
        #     rev(advantages) = discount_filter (X) rev(deltas)
        #
        # TensorFlow's tf.nn.conv1d op is not a true convolution, but
        # actually a cross-correlation, so its input and output are
        # already implicitly reversed for us.
        #    advantages = discount_filter (tf.nn.conv1d) deltas

        # Prepare convolutional IIR filter to calculate advantages
        gamma_lambda = tf.constant(
            float(discount) * float(gae_lambda),
            dtype=tf.float32,
            shape=[max_len, 1, 1])
        advantage_filter = tf.compat.v1.cumprod(gamma_lambda, exclusive=True)

        # Calculate deltas
        pad = tf.zeros_like(baselines[:, :1])
        baseline_shift = tf.concat([baselines[:, 1:], pad], 1)
        deltas = rewards + discount * baseline_shift - baselines
        # Convolve deltas with the discount filter to get advantages
        deltas_pad = tf.expand_dims(
            tf.concat([deltas, tf.zeros_like(deltas[:, :-1])], axis=1), axis=2)
        adv = tf.nn.conv1d(
            deltas_pad, advantage_filter, stride=1, padding='VALID')
        advantages = tf.reshape(adv, [-1])
    return advantages


def center_advs(advs, axes, eps, offset=0, scale=1, name=None):
    """ Normalize the advs tensor """
    with tf.name_scope(name, 'center_adv', [advs, axes, eps]):
        mean, var = tf.nn.moments(advs, axes=axes)
        advs = tf.nn.batch_normalization(advs, mean, var, offset, scale, eps)
    return advs


def positive_advs(advs, eps, name=None):
    """ Make all the values in the advs tensor positive """
    with tf.name_scope(name, 'positive_adv', [advs, eps]):
        m = tf.reduce_min(advs)
        advs = (advs - m) + eps
    return advs


def discounted_returns(discount, max_len, rewards, name=None):
    with tf.name_scope(name, 'discounted_returns',
                       [discount, max_len, rewards]):
        gamma = tf.constant(
            float(discount), dtype=tf.float32, shape=[max_len, 1, 1])
        return_filter = tf.math.cumprod(gamma, exclusive=True)
        rewards_pad = tf.expand_dims(
            tf.concat([rewards, tf.zeros_like(rewards[:, :-1])], axis=1),
            axis=2)
        returns = tf.nn.conv1d(
            rewards_pad, return_filter, stride=1, padding='VALID')
    return returns
