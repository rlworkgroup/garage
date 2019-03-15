"""`garage.logger` input types for logging distributions.

These are primarily used by `garage.logger.TensorBoardOutput`
"""

import numpy as np


class Empirical:
    """Represents a distribution empircally with a numerical dataset.

    The dataset may be histrogrammed to visualize and empirical distribution.

    EmpiricalDistribution may be passed to the logger via its log() method.

    :param data: The data to be turned into a histogram.
    """

    def __init__(self, data):
        self._data = data

    @property
    def data(self, ):
        """Log this data as a histogram."""
        return self._data


class Parametric:
    """Represents a distribution using its family and support parameters.

    ParametricDistribution may be passed to the logger via its log() method.

    :param family: The family of the distribution to be logged.
    :param shape: Shape of the output tensor.
    :param name: Alternate name for the distribution (default: `type(self)`)
    """

    def __init__(self, family, shape=None, name=None):
        self.family = family
        self.shape = shape
        self.name = name or type(self).__name__


class Normal(Parametric):
    """Represents a Normal distribution."""

    def __init__(self, shape=None, name=None, mean=None, stddev=None):
        super().__init__('normal', shape=shape, name=name)
        self.mean = _ensure_np_float_32(mean)
        self.stddev = _ensure_np_float_32(stddev)


class Gamma(Parametric):
    """Represents a Gamma distribution."""

    def __init__(self, shape=None, name=None, alpha=None):
        super().__init__('gamma', shape=shape, name=name)
        self.alpha = _ensure_np_float_32(alpha)


class Poisson(Parametric):
    """Represents a Poisson distribution."""

    def __init__(self, shape=None, name=None, lam=None):
        super().__init__('poisson', shape=shape, name=name)
        self.lam = _ensure_np_float_32(lam)


class Uniform(Parametric):
    """Represents a Uniform distribution."""

    def __init__(self, shape=None, name=None, maxval=None):
        super().__init__('uniform', shape=shape, name=name)
        self.maxval = _ensure_np_float_32(maxval)


def _ensure_np_float_32(val):
    """Cast to np.float32 if np.float64."""
    if isinstance(val, np.ndarray) and val.dtype == np.float64:
        return val.astype(np.float32)
    else:
        return val
