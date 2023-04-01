"""A garage optimizer that optimizes using a single large batch of SGD."""
import numpy as np

from garage import make_optimizer
from garage.torch.optimizers.optimizer import Optimizer


class SingleBatchOptimizer(Optimizer):
    """Optimizer that runs a torch.optim.Optimizer a single batch.

    Args:
        optimizer (Union[type, tuple[type, dict]]): Type of optimizer
            for policy. This can be an optimizer type such as
            `torch.optim.Adam` or a tuple of type and dictionary, where
            dictionary contains arguments to initialize the optimizer.
            e.g. `(torch.optim.Adam, {'lr' : 1e-3})`
            Sample strategy to be used when sampling a new task.
        module (torch.nn.Module): Module to be optimized.

    """

    def __init__(self, optimizer, module):
        super().__init__(module)
        self._optimizer = make_optimizer(optimizer, module=module)

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
        self._optimizer.zero_grad()
        loss = loss_function(**data)
        loss.backward()
        self._optimizer.step()
