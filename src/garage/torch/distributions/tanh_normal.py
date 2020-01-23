"""A Gaussian distribution with tanh transformation."""
from collections import namedtuple

import torch
from torch.distributions import Normal
from torch.distributions.independent import Independent


class TanhNormal(torch.distributions.Distribution):
    """A gaussian distribution that has a tanh transformation applied to it.

    Algorithms like SAC and Pearl use this transformed distribution.
    It can be thought of as a distribution of X where
        Y ~ N(mean, cov)
        X ~ tanh(Y)

    Args:
        mean (torch.tensor): The mean of this distribution.
        std (torch.tensor): The stdev of this distribution.

    """

    def __init__(self, mean, std):
        self._normal = Independent(Normal(mean, std), 1)
        super().__init__()

    def log_prob(self, value, pre_tanh_value=None, epsilon=1e-6):
        """The log likelihood of a sample on the this Tanh Distribution.

        Args:
            value (torch.Tensor): The sample whose loglikelihood is being
                computed.
            pre_tanh_value (torch.Tensor): The value prior to having the tanh
                function applied to it but after it has been sampled from the
                normal distribution.
            epsilon (float): Stabilization coefficient.

        Notes:
            - when pre_tanh_value is None, an estimate is made of what the
              value is. This leads to a worse estimation of the log_prob.
              If the value being used is collected from functions like
              `sample` and `rsample`, one can instead use functions like
              `sample_return_pre_tanh_value` or
              `rsample_return_pre_tanh_value`


        Returns:
            torch.Tensor: The log likelihood of value on the distribution.

        """
        # pylint: disable=arguments-differ
        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1 + value) / (1 - value)) / 2
        norm_lp = self._normal.log_prob(pre_tanh_value)
        ret = (norm_lp - torch.sum(
            torch.log(self._clip_but_pass_gradient((1. - value**2)) + epsilon),
            axis=-1))
        return ret

    def sample_return_pre_tanh_value(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this Tanh Normal Distribution.

        Returns the sampled value before the Tanh transform is applied and the
        sampled value with the Tanh transform applied to it.

        Args:
            sample_shape (list): Shape of the returned value.

        Note: Gradients `do not` pass through this operation.

        Returns:
            action_infos: named tuple with fields
                "pre_tanh_action" and "action"

        """
        with torch.no_grad():
            return self.rsample_return_pre_tanh_value(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this Tanh Normal Distribution.

        Args:
            sample_shape (list): Shape of the returned value.

        Note: Gradients pass through this operation.

        Returns:
            torch.Tensor: Sample from this Tanh Normal distribution.

        """
        z = self._normal.rsample(sample_shape=sample_shape)
        return torch.tanh(z)

    def rsample_return_pre_tanh_value(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this Tanh Normal distribution.

        Returns the sampled value before the Tanh transform is applied and the
        sampled value with the Tanh transform applied to it.

        Args:
            sample_shape (list): shape of the return.

        Note: Gradients pass through this operation.

        Returns:
            action_infos: named tuple with fields
                "pre_tanh_action" and "action"

        """
        action_infos = namedtuple('action_infos',
                                  ['pre_tanh_action', 'action'])
        z = self._normal.sample(sample_shape=sample_shape)
        return action_infos(z, torch.tanh(z))

    def cdf(self, value):
        """Returns the CDF at the value.

        Returns the cumulative density/mass function evaluated at
        `value` on the underlying normal distribution.

        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.

        Returns:
            torch.Tensor: the result of the cdf being computed.

        """
        return self._normal.cdf(value)

    def icdf(self, value):
        """Returns the icdf function evaluated at `value`.

        Returns the icdf function evaluated at `value` on the underlying
        normal distribution.

        Args:
            value (torch.Tensor): The element where the cdf is being evaluated
                at.

        Returns:
            torch.Tensor: the result of the cdf being computed.

        """
        return self._normal.icdf(value)

    @classmethod
    def _from_distribution(cls, new_normal):
        """Construct a new tanh distribution from a normal distribution.

        Args:
            new_normal (Independent(Normal)): underlying normal dist for
                the new TanhNormal dist.

        Returns:
            TanhNormal: A new distribution whose underlying normal dist
                is new_normal.

        """
        # pylint: disable=protected-access
        new = cls(torch.zeros(1), torch.zeros(1))
        new._normal = new_normal
        return new

    def expand(self, batch_shape, _instance=None):
        """Returns a new TanhNormal dist.

        (or populates an existing instance provided by a derived class) with
        batch dimensions expanded to `batch_shape`. This method calls
        :class:`~torch.Tensor.expand` on the distribution's parameters. As
        such, this does not allocate new memory for the expanded distribution
        instance. Additionally, this does not repeat any args checking or
        parameter broadcasting in `__init__.py`, when an instance is first
        created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance(instance): new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            Instance: New distribution instance with batch dimensions expanded
            to `batch_size`.

        """
        new_normal = self._normal.expand(batch_shape, _instance)
        new = self._from_distribution(new_normal)
        return new

    def enumerate_support(self, expand=True):
        """Returns tensor containing all values supported by a discrete dist.

        The result will enumerate over dimension 0, so the shape
        of the result will be `(cardinality,) + batch_shape + event_shape`
        (where `event_shape = ()` for univariate distributions).

        Note that this enumerates over all batched tensors in lock-step
        `[[0, 0], [1, 1], ...]`. With `expand=False`, enumeration happens
        along dim 0, but with the remaining batch dimensions being
        singleton dimensions, `[[0], [1], ..`.

        To iterate over the full Cartesian product use
        `itertools.product(m.enumerate_support())`.

        Args:
            expand (bool): whether to expand the support over the
                batch dims to match the distribution's `batch_shape`.

        Note:
            Calls the enumerate_support function of the underlying normal
            distribution.

        Returns:
            torch.Tensor: Tensor iterating over dimension 0.

        """
        return self._normal.enumerate_support(expand)

    @property
    def mean(self):
        """Returns mean of the distribution.

        Note: This is the mean of the underlying normal distribution
              with the Tanh transformation applied.

        Returns:
            torch.Tensor: mean of the distribution

        """
        return torch.tanh(self._normal.mean)

    @property
    def variance(self):
        """Returns variance of the underlying normal distribution.

        Returns:
            torch.Tensor: variance of the underlying normal distribution.

        """
        return self._normal.variance

    def entropy(self):
        """Returns entropy of the underlying normal distribution.

        Returns:
            torch.Tensor: entropy of the underlying normal distribution.

        """
        return self._normal.entropy()

    @staticmethod
    def _clip_but_pass_gradient(x, lower=0., upper=1.):
        """Clipping function that allows for gradients to flow through.

        Args:
            x(torch.tensor): value to be clipped
            lower(float): lower bound of clipping
            upper(float): upper bound of clipping

        Returns:
            torch.Tensor: x clipped between lower and upper.

        """
        clip_up = (x > upper).float()
        clip_low = (x < lower).float()
        with torch.no_grad():
            clip = ((upper - x) * clip_up + (lower - x) * clip_low)
        return x + clip

    def __str__(self):
        """Returns the Name of the class.

        Returns:
            str: The class name.

        """
        return 'TanhNormal'

    def __repr__(self):
        """Returns the parameterization of the distribution.

        Returns:
            str: The parameterization of the distribution and underlying
                distribution.

        """
        return ('TanhNormal Dist: Mean {}, stdev {}\n \
                 Underlying Normal Mean {}, stdev {}'.format(
            self.mean, self.variance, self._normal.mean,
            self._normal.variance))
