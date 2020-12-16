import abc


class Optimizer(metaclass=abc.ABCMeta):
    """Base class of optimizers in garage.torch.

    This class exists and differs from torch.optim.Optimizer for a few reasons:
        - Several optimizers (DifferentiableSGD, ConjugateGradientOptimizer)
          need access to the module they're optimizing as a whole, not just to
          their parameters.
        - The torch Optimizer class was not designed to be inherited from, and
          sometimes breaks base classes in difficult to detect ways.
        - The torch Optimizer API is large and not very easy to implement. If
          the whole API is needed, one of garage's Optimizer wrappers can be
          used instead.
        - We want our optimizer API to handle mini-batching, since it makes the
          relationship between PPO and TRPO simpler to implement and explain.

    Args:
        module (torch.nn.Module): The neural network to optimize.

    """

    def __init__(self, module):
        self._module = module

    @abc.abstractmethod
    def step(self, data, loss_function):
        """Use `data` to minimize `loss_function`.

        Note that data may be operated on in optimizer specific ways, and
        loss_function may be called multiple times.

        Args:
            data (dict[str, torch.Tensor]): Data to feed into the loss
                function. May be operated on before feeding.
            loss_function (dict[str, torch.Tensor] -> torch.Tensor):
                Differentiable loss function to optimize.

        Returns:
            float: Average value of loss_function over data.

        """
