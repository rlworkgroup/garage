"""Contains input classes for tensorboard.

Tensorboard accepts different types of inputs, so they are defined here to keep
tensorboard separate from the rest of the logger.
"""


class HistogramInput:
    """This class holds histogram information for TensorboardOutput.

    HistogramInput may be passed to the logger via its log() method.

    This type of histogram is generated using the given data.

    :param data: The data to be turned into a histogram.
    """

    def __init__(self, data):
        self._data = data

    @property
    def data(self, ):
        """Log this data as a histogram."""
        return self._data


class HistogramInputDistribution:
    """This class holds histogram information for TensorboardOutput.

    HistogramInput may be passed to the logger via its log() method.

    This type of histogram is built using the given type of distribution and
    its parameters.

    :param histogram_type: The class of histogram to be logged.
    :param shape: Shape of the output tensor.
    :param name: Alternate name for histogram instead of simply the type.
    """

    def __init__(self, histogram_type, shape=None, name=None):
        self.histogram_type = histogram_type
        self.shape = shape
        self.name = name


class HistogramInputNormal(HistogramInputDistribution):
    """Builds a histogram using the Normal distribution."""

    def __init__(self, shape=None, name=None, mean=None, stddev=None):
        super().__init__("normal", shape=shape, name=name)
        self.mean = mean
        self.stddev = stddev


class HistogramInputGamma(HistogramInputDistribution):
    """Builds a histogram using the Gamma distribution."""

    def __init__(self, shape=None, name=None, alpha=None):
        super().__init__("gamma", shape=shape, name=name)
        self.alpha = alpha


class HistogramInputPoisson(HistogramInputDistribution):
    """Builds a histogram using the Poisson distribution."""

    def __init__(self, shape=None, name=None, lam=None):
        super().__init__("poisson", shape=shape, name=name)
        self.lam = lam


class HistogramInputUniform(HistogramInputDistribution):
    """Builds a histogram using the Uniform distribution."""

    def __init__(self, shape=None, name=None, maxval=None):
        super().__init__("uniform", shape=shape, name=name)
        self.maxval = maxval
