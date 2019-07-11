"""Utilities for ensuring that experiments are deterministic."""
import random
import sys

import numpy as np

seed_ = None


def set_seed(seed):
    """Set the process-wide random seed.

    Args:
        seed (int): A positive integer

    """
    seed %= 4294967294
    global seed_
    seed_ = seed
    random.seed(seed)
    np.random.seed(seed)
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.compat.v1.set_random_seed(seed)


def get_seed():
    """Get the process-wide random seed.

    Returns:
        int: The process-wide random seed

    """
    return seed_
