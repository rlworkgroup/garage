import numpy as np
import torch
from torch.distributions import constraints
from torch.distributions import MultivariateNormal
from torch.distributions import Transform
from torch.distributions.transformed_distribution import TransformedDistribution
import torch.nn.functional as F

class TanhTransform(Transform):
    """ Tanh transformation for torch distributions."""
    domain = constraints.real
    codomain = constraints.interval(-1.,1.)
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh().clamp(-1. + 3e-6, 1. - 3e-6)

    def _inverse(self, y):
        atanh_val = torch.log((1+y) / (1-y)) / 2
        if(torch.isnan(atanh_val).any()):
            import ipdb; ipdb.set_trace()
        return atanh_val

    def log_prob(self, value):
        log_prob = super.log_prob(value)
        return log_prob

    def log_abs_det_jacobian(self, x, y):
        return  2. * (np.log(2.) - x - F.softplus(-2. * x))

def _get_checked_instance(self, cls, _instance=None):
    if _instance is None and type(self).__init__ != cls.__init__:
        raise NotImplementedError("Subclass {} of {} that defines a custom __init__ method "
                                    "must also define a custom .expand() method.".
                                    format(self.__class__.__name__, cls.__name__))
    return self.__new__(type(self)) if _instance is None else _instance


class TanhNormal(TransformedDistribution):
    """ A Multivariate Gaussian Distribution with tanh output.

    Args: 
        loc (Tensor): mean of the distribution
        covariance_matrix (Tensor): positive-definite covariance matrix
        precision_matrix (Tensor): positive-definite precision matrix
        scale_tril (Tensor): lower-triangular factor of covariance, with positive-valued diagonal 
    """
    arg_constraints = {'loc': constraints.real_vector,
                       'covariance_matrix': constraints.positive_definite,
                       'precision_matrix': constraints.positive_definite,
                       'scale_tril': constraints.lower_cholesky}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, covariance_matrix=None, validate_args=None):
        try:
            self.base_dist = MultivariateNormal(loc, covariance_matrix)
        except Exception as e:
            import ipdb; ipdb.set_trace()
            print(e)
        super(TanhNormal, self).__init__(self.base_dist, TanhTransform(), validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TanhNormal, _instance)
        return super(TanhNormal, self).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        return torch.tanh(self.base_dist.mean)

    @property
    def variance(self):
        return self.base_dist.variance

    def entropy(self):
        return self.base_dist.entropy()
