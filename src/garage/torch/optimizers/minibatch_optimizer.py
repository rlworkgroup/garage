"""A garage optimizer that optimizes using minibatches."""
import click
import numpy as np

from garage import make_optimizer
from garage.torch.optimizers.optimizer import Optimizer


class MinibatchOptimizer(Optimizer):
    """Optimizer that runs a torch.optim.Optimizer on minibatches.

    Args:
        optimizer (Union[type, tuple[type, dict]]): Type of optimizer
            for policy. This can be an optimizer type such as
            `torch.optim.Adam` or a tuple of type and dictionary, where
            dictionary contains arguments to initialize the optimizer.
            e.g. `(torch.optim.Adam, {'lr' : 1e-3})`
            Sample strategy to be used when sampling a new task.
        module (torch.nn.Module): Module to be optimized.
        max_optimization_epochs (int): Maximum number of times to iterate
            through all samples.
        minibatch_size (int): Batch size for optimization. If a single large
            batch is desired, consider using SingleBatchOptimizer instead.

    """

    def __init__(self,
                 optimizer,
                 module,
                 max_optimization_epochs=1,
                 minibatch_size=32):
        super().__init__(module)
        self._optimizer = make_optimizer(optimizer, module=module)
        self._max_optimization_epochs = max_optimization_epochs
        self._minibatch_size = minibatch_size

    def _minibatches(self, n_samples, data):
        r"""Yields a batch of inputs.

        Notes: P is the size of minibatch (self._minibatch_size)

        Args:
            n_samples (int): Total number of samples in data.
            data (dict[str, torch.Tensor]): Data to sample into batches. Each
                tensor has shape :math:`(N \dot [T], *)`.

        Yields:
            dict[str, torch.Tensor]: Batch of inputs to pass to loss function.

        """
        assert n_samples == len(next(iter(data.values())))
        with click.progressbar(range(self._max_optimization_epochs),
                               label='Optimizing') as pbar:
            for _ in pbar:
                all_indices = np.arange(n_samples)
                np.random.shuffle(all_indices)
                split = np.array_split(
                    all_indices, np.ceil(n_samples / self._minibatch_size))
                for minibatch_indices in split:
                    yield {k: v[minibatch_indices] for (k, v) in data.items()}

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
        if 'lengths' in data:
            del data['lengths']
        n_samples = [len(v) for v in data.values()]
        assert all(n == n_samples[0] for n in n_samples)

        for i, batch in enumerate(self._minibatches(n_samples[0], data)):
            self._optimizer.zero_grad()
            loss = loss_function(**batch)
            loss.backward()
            self._optimizer.step()
