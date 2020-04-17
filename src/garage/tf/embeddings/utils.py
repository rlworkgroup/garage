"""Utilities for embedding."""
import akro
import numpy as np


def concat_spaces(first, second):
    """Concatenate two Box space.

    Args:
        first (akro.Box): The first space.
        second (akro.Box): The second space.

    Returns:
        akro.Box: The concatenated space.

    """
    assert isinstance(first, akro.Box)
    assert isinstance(second, akro.Box)

    first_lb, first_ub = first.bounds
    second_lb, second_ub = second.bounds
    return akro.Box(np.concatenate([first_lb, second_lb]),
                    np.concatenate([first_ub, second_ub]))
