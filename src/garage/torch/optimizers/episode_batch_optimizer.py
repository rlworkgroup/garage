"""Optimizer that runs a torch optimizer on full episodes."""
import click
import numpy as np

from garage import make_optimizer
from garage.torch import (as_tensor, ObservationBatch, ObservationOrder,
                          split_packed_tensor)
from garage.torch.optimizers.optimizer import Optimizer


class EpisodeBatchOptimizer(Optimizer):
    """Optimizer that runs a torch optimizer on full episodes.

    Args:
        optimizer (Union[type, tuple[type, dict]]): Type of optimizer
            for policy. This can be an optimizer type such as
            `torch.optim.Adam` or a tuple of type and dictionary, where
            dictionary contains arguments to initialize the optimizer.
            e.g. `(torch.optim.Adam, {'lr' : 1e-3})`
            Sample strategy to be used when sampling a new task.
        module (torch.nn.Module): Module to be optimized.
        max_optimization_epochs (int): Maximum number of epochs for update.
        minibatch_size (int): Batch size for optimization.

    """

    def __init__(self,
                 optimizer,
                 module,
                 max_optimization_epochs=1000,
                 minibatch_size=32):
        super().__init__(module)
        self._optimizer = make_optimizer(optimizer, module=module)
        self._max_optimization_epochs = max_optimization_epochs
        self._minibatch_size = minibatch_size

    def _minibatches(self, data_by_episode, lengths):
        r"""Yields a batch of inputs.

        Notes: P is the size of minibatch (self._minibatch_size)

        Args:
            data_by_episode (dict[str, list[torch.Tensor]]): Dictionary of
                data, where each data array has been split by episode.
            lengths (list[int]): Length of each episode in data.

        Yields:
            dict[str, torch.Tensor]: Batch of inputs to pass to loss function.

        """
        episode_indices = np.range(len(lengths))
        i = 0
        with click.progressbar(range(self._max_optimization_epochs),
                               label='Optimizing') as pbar:
            for _ in pbar:
                batch_size = 0
                batch = {k: [] for k in data_by_episode.keys()}
                batch_lengths = []
                while sum(batch_lengths) < self._minibatch_size:
                    if i == 0:
                        np.random.shuffle(episode_indices)
                    for k, v in data_by_episode.items():
                        batch[k].append(v[i])
                    batch_lengths.append(lengths[i])
                    i = (i + 1) % len(lengths)
                batch = {k: as_tensor(v) for (k, v) in batch.items()}
                batch['observations'] = ObservationBatch(
                    batch['observations'], ObservationOrder.EPISODES,
                    batch_lengths)
                batch['lengths'] = as_tensor(batch_lengths)
                yield batch

    def step(self, data, loss_function):
        """Use `data` to minimize `loss_function`.

        Note that data may be operated on in optimizer specific ways, and
        loss_function may be called multiple times.

        Args:
            data (dict[str, torch.Tensor]): Data to feed into the loss
                function. May be operated on before feeding. Must contain the
                key 'lengths'.
            loss_function (dict[str, torch.Tensor] -> torch.Tensor):
                Differentiable loss function to optimize.

        Returns:
            float: Average value of loss_function over data.

        """
        if 'observations' not in data:
            raise ValueError('observations must be in data for '
                             'EpisodeBatchOptimizer')
        try:
            lengths = data['lengths']
        except KeyError:
            try:
                lengths = data['observations'].lengths
            except AttributeError:
                raise ValueError('EpisodeBatchOptimizer must have lengths in '
                                 'data or observations must be an '
                                 'ObservationBatch')
        data_by_episode = {
            k: split_packed_tensor(v, lengths)
            for (k, v) in data.items() if v != 'lengths'
        }
        for batch in self._minibatches(data_by_episode, lengths):
            self._optimizer.zero_grad()
            loss = loss_function(**batch)
            loss.backward()
            self._optimizer.step()
