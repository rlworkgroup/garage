from functools import reduce
import operator
import random
import sys

import numpy as np

sys.setrecursionlimit(50000)


def extract(x, *keys):
    if isinstance(x, (dict, LazyDict)):
        return tuple(x[k] for k in keys)
    elif isinstance(x, list):
        return tuple([xi[k] for xi in x] for k in keys)
    else:
        raise NotImplementedError


def extract_dict(x, *keys):
    return {k: x[k] for k in keys if k in x}


def flatten(xs):
    return [x for y in xs for x in y]


def compact(x):
    """
    For a dictionary this removes all None values, and for a list this removes
    all None elements; otherwise it returns the input itself.
    """
    if isinstance(x, dict):
        return dict((k, v) for k, v in x.items() if v is not None)
    elif isinstance(x, list):
        return [elem for elem in x if elem is not None]
    return x


# Immutable, lazily evaluated dict
class LazyDict:
    def __init__(self, **kwargs):
        self._lazy_dict = kwargs
        self._dict = {}

    def __getitem__(self, key):
        if key not in self._dict:
            self._dict[key] = self._lazy_dict[key]()
        return self._dict[key]

    def __setitem__(self, i, y):
        self.set(i, y)

    def get(self, key, default=None):
        if key in self._lazy_dict:
            return self[key]
        return default

    def set(self, key, value):
        self._lazy_dict[key] = value


def iscanl(f, l, base=None):
    started = False
    for x in l:
        if base or started:
            base = f(base, x)
        else:
            base = x
        started = True
        yield base


def iscanr(f, l, base=None):
    started = False
    for x in list(l)[::-1]:
        if base or started:
            base = f(x, base)
        else:
            base = x
        started = True
        yield base


def scanl(f, l, base=None):
    return list(iscanl(f, l, base))


def scanr(f, l, base=None):
    return list(iscanr(f, l, base))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def is_iterable(obj):
    return isinstance(obj, str) or getattr(obj, '__iter__', False)


# cut the path for any time >= t
def truncate_path(p, t):
    return dict((k, p[k][:t]) for k in p)


def concat_paths(p1, p2):
    import numpy as np
    return dict((k1, np.concatenate([p1[k1], p2[k1]]))
                for k1 in list(p1.keys()) if k1 in p2)


def path_len(p):
    return len(p["states"])


def shuffled(sequence):
    deck = list(sequence)
    while deck:
        i = random.randint(0, len(deck) - 1)  # choose random card
        card = deck[i]  # take the card
        deck[i] = deck[-1]  # put top card in its place
        deck.pop()  # remove top card
        yield card


seed_ = None


def set_seed(seed):
    seed %= 4294967294
    global seed_
    seed_ = seed
    random.seed(seed)
    np.random.seed(seed)
    try:
        if "tensorflow" in sys.modules:
            import tensorflow as tf
            tf.set_random_seed(seed)
    except Exception as e:
        print(e)


def get_seed():
    return seed_


def flatten_shape_dim(shape):
    return reduce(operator.mul, shape, 1)


def print_lasagne_layer(layer, prefix=""):
    params = ""
    if layer.name:
        params += ", name=" + layer.name
    if getattr(layer, 'nonlinearity', None):
        params += ", nonlinearity=" + layer.nonlinearity.__name__
    params = params[2:]
    print(prefix + layer.__class__.__name__ + "[" + params + "]")
    if hasattr(layer, 'input_layers') and layer.input_layers is not None:
        [print_lasagne_layer(x, prefix + "  ") for x in layer.input_layers]
    elif hasattr(layer, 'input_layer') and layer.input_layer is not None:
        print_lasagne_layer(layer.input_layer, prefix + "  ")


def sliced_fun(f, n_slices):
    """Devide function f's inputs into several slices.

    Evaluate f on those slices, and then average the result. It is useful when
    memory is not enough to process all data at once.
    Assume:
    1. each of f's inputs is iterable and composed of multiple "samples"
    2. outputs can be averaged over "samples"
    """

    def sliced_f(sliced_inputs, non_sliced_inputs=None):
        if non_sliced_inputs is None:
            non_sliced_inputs = []
        if isinstance(non_sliced_inputs, tuple):
            non_sliced_inputs = list(non_sliced_inputs)
        n_paths = len(sliced_inputs[0])
        slice_size = max(1, n_paths // n_slices)
        ret_vals = None
        for start in range(0, n_paths, slice_size):
            inputs_slice = [v[start:start + slice_size] for v in sliced_inputs]
            slice_ret_vals = f(*(inputs_slice + non_sliced_inputs))
            if not isinstance(slice_ret_vals, (tuple, list)):
                slice_ret_vals_as_list = [slice_ret_vals]
            else:
                slice_ret_vals_as_list = slice_ret_vals
            scaled_ret_vals = [
                np.asarray(v) * len(inputs_slice[0])
                for v in slice_ret_vals_as_list
            ]
            if ret_vals is None:
                ret_vals = scaled_ret_vals
            else:
                ret_vals = [x + y for x, y in zip(ret_vals, scaled_ret_vals)]
        ret_vals = [v / n_paths for v in ret_vals]
        if not isinstance(slice_ret_vals, (tuple, list)):
            ret_vals = ret_vals[0]
        elif isinstance(slice_ret_vals, tuple):
            ret_vals = tuple(ret_vals)
        return ret_vals

    return sliced_f


def stdize(data, eps=1e-6):
    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + eps)


def iterate_minibatches_generic(input_lst=None, batchsize=None, shuffle=False):
    if batchsize is None:
        batchsize = len(input_lst[0])

    assert all(len(x) == len(input_lst[0]) for x in input_lst)

    if shuffle:
        indices = np.arange(len(input_lst[0]))
        np.random.shuffle(indices)
    for start_idx in range(0, len(input_lst[0]), batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in input_lst]
