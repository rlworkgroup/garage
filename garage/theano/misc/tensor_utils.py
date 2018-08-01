"""Utilities for Theano tensors."""
from collections import OrderedDict
import os.path as osp
import pickle

import numpy as np
import theano
from theano import grad
from theano import Variable
from theano.gradient import format_as
import theano.tensor as TT
from theano.tensor import arange
import theano.tensor.extra_ops
import theano.tensor.nnet

from garage.misc.console import Message


def softmax_sym(x):
    """Return the softmax function of x."""
    return theano.tensor.nnet.softmax(x)


def normalize_updates(old_mean, old_std, new_mean, new_std, old_w, old_b):
    """Compute the updates to normalize the last layer of a neural network."""
    # Make necessary transformation so that
    # (W_old * h + b_old) * std_old + mean_old == \
    #   (W_new * h + b_new) * std_new + mean_new
    new_w = old_w * old_std[0] / (new_std[0] + 1e-6)
    new_b = (old_b * old_std[0] + old_mean[0] - new_mean[0]) / (
        new_std[0] + 1e-6)
    return OrderedDict([
        (old_w, TT.cast(new_w, old_w.dtype)),
        (old_b, TT.cast(new_b, old_b.dtype)),
        (old_mean, new_mean),
        (old_std, new_std),
    ])


def to_onehot_sym(ind, dim):
    """Return a matrix with one hot encoding of each element in ind."""
    assert ind.ndim == 1
    return theano.tensor.extra_ops.to_one_hot(ind, dim)


def cached_function(inputs, outputs):
    """Find and load a file with a cached tensor function.

    Returns
    -------
    A callable object that will calculate outputs from inputs.

    """
    with Message("Hashing theano fn"):
        if hasattr(outputs, '__len__'):
            hash_content = tuple(map(theano.pp, outputs))
        else:
            hash_content = theano.pp(outputs)
    cache_key = hex(hash(hash_content) & (2**64 - 1))[:-1]
    cache_dir = osp.expanduser('~/.hierctrl_cache')
    cache_file = cache_dir + ("/%s.pkl" % cache_key)
    if osp.isfile(cache_file):
        with Message("unpickling"):
            with open(cache_file, "rb") as f:
                try:
                    return pickle.load(f)
                except Exception:
                    pass
    with Message("compiling"):
        fun = compile_function(inputs, outputs)
    with Message("picking"):
        with open(cache_file, "wb") as f:
            pickle.dump(fun, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fun


def compile_function(inputs=None,
                     outputs=None,
                     updates=None,
                     givens=None,
                     log_name=None,
                     **kwargs):
    """Return a callable object that will calculate outputs from inputs."""
    if log_name:
        msg = Message("Compiling function %s" % log_name)
        msg.__enter__()
    ret = theano.function(
        inputs=inputs,
        outputs=outputs,
        updates=updates,
        givens=givens,
        on_unused_input='ignore',
        allow_input_downcast=True,
        **kwargs)
    if log_name:
        msg.__exit__(None, None, None)
    return ret


def new_tensor(name, ndim, dtype):
    """Return a new tensor based on the data type and name provided."""
    return TT.TensorType(dtype, (False, ) * ndim)(name)


def new_tensor_like(name, arr_like):
    """Return a new tensor based on arr_like."""
    return new_tensor(name, arr_like.ndim, arr_like.dtype)


def flatten_hessian(cost,
                    wrt,
                    consider_constant=None,
                    disconnected_inputs='raise',
                    block_diagonal=True):
    """Flatten hessian.

    :type cost: Scalar (0-dimensional) Variable.
    :type wrt: Vector (1-dimensional tensor) 'Variable' or list of
               vectors (1-dimensional tensors) Variables

    :param consider_constant: a list of expressions not to backpropagate
        through

    :type disconnected_inputs: string
    :param disconnected_inputs: Defines the behaviour if some of the variables
        in ``wrt`` are not part of the computational graph computing ``cost``
        (or if all links are non-differentiable). The possible values are:
        - 'ignore': considers that the gradient on these parameters is zero.
        - 'warn': consider the gradient zero, and print a warning.
        - 'raise': raise an exception.

    :return: either a instance of Variable or list/tuple of Variables
            (depending upon `wrt`) repressenting the Hessian of the `cost`
            with respect to (elements of) `wrt`. If an element of `wrt` is not
            differentiable with respect to the output, then a zero
            variable is returned. The return value is of same type
            as `wrt`: a list/tuple or TensorVariable in all cases.
    """
    assert isinstance(cost, Variable), \
        "tensor.hessian expects a Variable as `cost`"
    assert cost.ndim == 0, \
        "tensor.hessian expects a 0 dimensional variable as `cost`"

    using_list = isinstance(wrt, list)
    using_tuple = isinstance(wrt, tuple)

    if isinstance(wrt, (list, tuple)):
        wrt = list(wrt)
    else:
        wrt = [wrt]

    hessians = []
    if not block_diagonal:
        expr = TT.concatenate([
            grad(
                cost,
                input,
                consider_constant=consider_constant,
                disconnected_inputs=disconnected_inputs).flatten()
            for input in wrt
        ])

    for input in wrt:
        assert isinstance(input, Variable), \
            "tensor.hessian expects a (list of) Variable as `wrt`"
        # assert input.ndim == 1, \
        #     "tensor.hessian expects a (list of) 1 dimensional variable " \
        #     "as `wrt`"
        if block_diagonal:
            expr = grad(
                cost,
                input,
                consider_constant=consider_constant,
                disconnected_inputs=disconnected_inputs).flatten()

        # It is possible that the inputs are disconnected from expr,
        # even if they are connected to cost.
        # This should not be an error.
        hess, updates = theano.scan(
            lambda i, y, x: grad(
                y[i],
                x,
                consider_constant=consider_constant,
                disconnected_inputs='ignore').flatten(),
            sequences=arange(expr.shape[0]),
            non_sequences=[expr, input])
        assert not updates, \
            ("Scan has returned a list of updates. This should not "
             "happen! Report this to theano-users (also include the "
             "script that generated the error)")
        hessians.append(hess)
    if block_diagonal:
        return format_as(using_list, using_tuple, hessians)
    else:
        return TT.concatenate(hessians, axis=1)


def flatten_tensor_variables(ts):
    """Return all tensors in ts as a flat tensor of dimension one."""
    return TT.concatenate(list(map(TT.flatten, ts)))


def unflatten_tensor_variables(flatarr, shapes, symb_arrs):
    """Reshape the flat array into the given shapes."""
    arrs = []
    n = 0
    for (shape, symb_arr) in zip(shapes, symb_arrs):
        size = np.prod(list(shape))
        arr = flatarr[n:n + size].reshape(shape)
        if arr.type.broadcastable != symb_arr.type.broadcastable:
            arr = TT.patternbroadcast(arr, symb_arr.type.broadcastable)
        arrs.append(arr)
        n += size
    return arrs
