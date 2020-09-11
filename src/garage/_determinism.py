"""Utilities for ensuring that experiments are deterministic."""
import random
import sys

from dowel import logger
import numpy as np

__seed = random.SystemRandom().randrange(1, sys.maxsize)
__seed_stream = None


def set_seed(seed):
    """Set the process-wide random seed.

    Args:
        seed (int): A positive integer

    """
    # pylint: disable=global-statement
    global __seed
    global __seed_stream

    seed %= 4294967294
    __seed = seed
    random.seed(__seed)
    np.random.seed(__seed)

    if 'tensorflow' in sys.modules:
        # pylint: disable=import-outside-toplevel
        import tensorflow as tf
        tf.compat.v1.set_random_seed(__seed)

    if 'torch' in sys.modules:
        # pylint: disable=import-outside-toplevel
        import torch
        logger.log(
            'Enabling deterministic mode in PyTorch can have a performance '
            'impact when using GPU.')
        torch.manual_seed(__seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed():
    """Get the process-wide random seed.

    Returns:
        int: The process-wide random seed

    """
    return __seed


def get_tf_seed_stream():
    """Get the pseudo-random number generator (PRNG) for TensorFlow ops.

    Returns:
        tensorflow_probability.util.SeedStream: Process-global sequence of
           random numbers with a fixed seed.

    Raises:
        RuntimeError: If tensorflow_probability is not installed.

    """
    # pylint: disable=global-statement
    global __seed_stream

    if __seed_stream is None:
        # pylint: disable=import-outside-toplevel
        if 'tensorflow_probability' not in sys.modules:
            raise RuntimeError('You must install tensorflow and '
                               'tensorflow_probability to use this API.')

        import tensorflow_probability as tfp
        __seed_stream = tfp.util.SeedStream(__seed, salt='garage')

    return __seed_stream
