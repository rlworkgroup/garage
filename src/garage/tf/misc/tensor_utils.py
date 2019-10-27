"""Utility functions for tensor in TensorFlow."""
from collections import Iterable
from collections import namedtuple

import numpy as np
import tensorflow as tf


def compile_function(inputs, outputs):
    """Compile function.

    Args:
        inputs (list[tf.Tensor]): List of input tensors.
        outputs (list[tf.Tensor]): List of output tensors.

    Returns:
        callable: Compiled function call.

    """

    def run(*input_vals):
        """Output tensors when executing a session.

        Args:
            input_vals (list[numpy.ndarray]): Data fed to tensors.

        Returns:
            list[numpy.ndarray]: Outputs.

        """
        sess = tf.compat.v1.get_default_session()
        return sess.run(outputs, feed_dict=dict(list(zip(inputs, input_vals))))

    return run


def get_target_ops(variables, target_variables, tau=None):
    """Get target variables update operations.

    In RL algorithms we often update target network every n
    steps. This function returns the tf.Operation for updating
    target variables (denoted by target_var) from variables
    (denote by var) with fraction tau. In other words, each time
    we want to keep tau of the var and add (1 - tau) of target_var
    to var.

    Args:
        variables (list[tf.Variable]): Source variables for update.
        target_variables (list[tf.Variable]): Target variables to
            be updated.
        tau (float): Fraction to update. Set it to be None for
            hard-update.

    Returns:
        list[tf.Operations]: Target operations.

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
    return init_ops


def flatten_batch_tensor(t, name='flatten_batch_tensor'):
    """Flatten input along batch dimension.

    Args:
        t (tf.Tensor): Input tensor to be flattened.
        name (str): Operation name.

    Returns:
        tf.Tensor: Flattened input.

    """
    return tf.reshape(t, [-1] + list(t.shape[2:]), name=name)


def flatten_batch_tensor_dict(d, name=None):
    """Flatten input along batch dimension.

    Args:
        d (dict(tf.Tensor)): Input tensor to be flattened.
        name (str): Operation name.

    Returns:
        dict(tf.Tensor): Flattened input.

    """
    with tf.name_scope(name, 'flatten_batch_tensor_dict', [d]):
        return {k: flatten_batch_tensor(v) for k, v in d.items()}


def filter_valids_tensor(t, valid, name='filter_valids_tensor'):
    """Filter input with valids.

    Args:
        t (tf.Tensor): Input tensor to be flattened.
        valid (tf.Tensor): Tensor with element either 0 or 1. The value of
            i-th element being 1 indicates i-th index is valid.
        name (str): Operation name.

    Returns:
        tf.Tensor: Flattened input.

    """
    # 'valid' is either 0 or 1 with dtype of tf.float32
    # Must round before cast to prevent floating-error
    return tf.dynamic_partition(t,
                                tf.cast(tf.round(valid), tf.int32),
                                2,
                                name=name)[1]


def filter_valids_tensor_dict(d, valid, name=None):
    """Filter input with valids.

    Args:
        d (dict(tf.Tensor)): Input tensor to be flattened.
        valid (tf.Tensor): Tensor with element either 0 or 1. The value of
            i-th element being 1 indicates i-th index is valid.
        name (str): Operation name.

    Returns:
        dict(tf.Tensor): Flattened input.

    """
    with tf.name_scope(name, 'filter_valids_tensor_dict', [d, valid]):
        return {k: filter_valids_tensor(v, valid) for k, v in d.items()}


def graph_inputs(name, **kwargs):
    """Gather tf.Tensors as graph inputs.

    Args:
        name (str): Name of the group of graph inputs.
        kwargs (dict): Key and value pairs of the inputs.

    Returns:
        namedtuple: Group of graph inputs.

    """
    Singleton = namedtuple(name, kwargs.keys())
    return Singleton(**kwargs)


def flatten_inputs(deep):
    """Flatten inputs.

    Args:
        deep (list): Inputs to be flattened.

    Returns:
        List: Flattened input.

    """

    def flatten(deep):
        """Flatten.

        Args:
            deep (list): Inputs to be flattened.

        Yields:
            List: Flattened input.

        """
        for d in deep:
            if isinstance(d, Iterable) and not isinstance(
                    d, (str, bytes, tf.Tensor, np.ndarray)):
                yield from flatten(d)
            else:
                yield d

    return list(flatten(deep))


def flatten_tensor_variables(ts):
    """Flatten tensor variables.

    Args:
        ts (tf.Tensor): Input to be flattened.

    Returns:
        tf.Tensor: Flattened input.

    """
    return tf.concat(axis=0,
                     values=[tf.reshape(x, [-1]) for x in ts],
                     name='flatten_tensor_variables')


def new_tensor(name, ndim, dtype):
    """Create a new tensor.

    Args:
        name (str): Name of the tensor.
        ndim (int): Dimension of the tensor.
        dtype (tf.dtype): Data type of the tensor.

    Returns:
        tf.Tensor: The new tensor.

    """
    return tf.compat.v1.placeholder(dtype=dtype,
                                    shape=[None] * ndim,
                                    name=name)


def new_tensor_like(name, arr_like):
    """Create a new tensor with same shape and dtype as input.

    Args:
        name (str): Name of the tensor.
        arr_like (tf.Tensor): Targe tensor.

    Returns:
        tf.Tensor: The new tensor.

    """
    return new_tensor(name,
                      arr_like.get_shape().ndims, arr_like.dtype.base_dtype)


def compute_advantages(discount,
                       gae_lambda,
                       max_len,
                       baselines,
                       rewards,
                       name=None):
    """Calculate advantages.

    Advantages are a discounted cumulative sum.

    The discount cumulative sum can be represented as an IIR
    filter ob the reversed input vectors, i.e.
       y[t] - discount*y[t+1] = x[t]
           or
       rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]

    Given the time-domain IIR filter step response, we can
    calculate the filter response to our signal by convolving the
    signal with the filter response function. The time-domain IIR
    step response is calculated below as discount_filter:
        discount_filter =
            [1, discount, discount^2, ..., discount^N-1]
            where the epsiode length is N.

    We convolve discount_filter with the reversed time-domain
    signal deltas to calculate the reversed advantages:
        rev(advantages) = discount_filter (X) rev(deltas)

    TensorFlow's tf.nn.conv1d op is not a true convolution, but
    actually a cross-correlation, so its input and output are
    already implicitly reversed for us.
       advantages = discount_filter (tf.nn.conv1d) deltas

    Args:
        discount (float): Discount factor.
        gae_lambda (float): GAE labmda parameter.
        max_len (int): Maximum path length.
        baselines (tf.Tensor): Baselines.
        rewards (tf.Tensor): Rewards.
        name (str): Name scope.

    Returns:
        tf.Tensor: Advantage.

    """
    with tf.name_scope(name, 'compute_advantages',
                       [discount, gae_lambda, max_len, baselines, rewards]):
        # Prepare convolutional IIR filter to calculate advantages
        gamma_lambda = tf.constant(float(discount) * float(gae_lambda),
                                   dtype=tf.float32,
                                   shape=[max_len, 1, 1])
        advantage_filter = tf.compat.v1.cumprod(gamma_lambda, exclusive=True)

        # Calculate deltas
        pad = tf.zeros_like(baselines[:, :1])
        baseline_shift = tf.concat([baselines[:, 1:], pad], 1)
        deltas = rewards + discount * baseline_shift - baselines
        # Convolve deltas with the discount filter to get advantages
        deltas_pad = tf.expand_dims(tf.concat(
            [deltas, tf.zeros_like(deltas[:, :-1])], axis=1),
                                    axis=2)
        adv = tf.nn.conv1d(deltas_pad,
                           advantage_filter,
                           stride=1,
                           padding='VALID')
        advantages = tf.reshape(adv, [-1])
    return advantages


def center_advs(advs, axes, eps, offset=0, scale=1, name=None):
    """Normalize the advs tensor.

    Args:
        advs (tf.Tensor): Advantage tensor.
        axes (list[int]): Axes to normalize along.
        eps (tf.Tensor): Variable epsilon. A small float number to avoid
            dividing by 0.
        offset (tf.Tensor): An offset Tensor. If present, will be added to
            the normalized tensor.
        scale (tf.Tensor): A scale Tensor, If present, the scale is applied
            to the normalized tensor.
        name (str): Name scope.

    Returns:
        tf.Tensor: Normalized advantage.

    """
    with tf.name_scope(name, 'center_adv', [advs, axes, eps]):
        mean, var = tf.nn.moments(advs, axes=axes)
        advs = tf.nn.batch_normalization(advs, mean, var, offset, scale, eps)
    return advs


def positive_advs(advs, eps, name=None):
    """Make all the values in the advs tensor positive.

    Args:
        advs (tf.Tensor): Advantage tensor.
        eps (tf.Tensor): Variable epsilon. A small float number to avoid
            dividing by 0.
        name (str): Name scope.

    Returns:
        tf.Tensor: Positive advantage.

    """
    with tf.name_scope(name, 'positive_adv', [advs, eps]):
        m = tf.reduce_min(advs)
        advs = (advs - m) + eps
    return advs


def discounted_returns(discount, max_len, rewards, name=None):
    """Discounted returns.

    Args:
        discount (float): Discount factor.
        max_len (int): Maximum path length.
        rewards (tf.Tensor): Rewards tensor.
        name (str): Name scope.

    Returns:
        tf.Tensor: Discounted returns.

    """
    with tf.name_scope(name, 'discounted_returns',
                       [discount, max_len, rewards]):
        gamma = tf.constant(float(discount),
                            dtype=tf.float32,
                            shape=[max_len, 1, 1])
        return_filter = tf.math.cumprod(gamma, exclusive=True)
        rewards_pad = tf.expand_dims(tf.concat(
            [rewards, tf.zeros_like(rewards[:, :-1])], axis=1),
                                     axis=2)
        returns = tf.nn.conv1d(rewards_pad,
                               return_filter,
                               stride=1,
                               padding='VALID')
    return returns
