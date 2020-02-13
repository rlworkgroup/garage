import akro
import numpy as np


def concat_spaces(top, bottom):
    assert isinstance(top, akro.Box)
    assert isinstance(bottom, akro.Box)

    top_lb, top_ub = top.bounds
    bottom_lb, bottom_ub = bottom.bounds
    return akro.Box(
        np.concatenate([top_lb, bottom_lb]),
        np.concatenate([top_ub, bottom_ub]))
