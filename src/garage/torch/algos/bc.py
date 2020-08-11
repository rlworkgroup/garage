"""Implementation of Behavioral Cloning in PyTorch."""
# yapf: disable
import itertools

from dowel import tabular
import numpy as np
import torch

from garage import (_Default,
                    EpisodeBatch,
                    log_performance,
                    make_optimizer,
                    TimeStepBatch)
from garage.np import obtain_evaluation_episodes
from garage.np.algos.rl_algorithm import RLAlgorithm
from garage.np.policies import Policy
from garage.sampler import RaySampler
from garage.torch import np_to_torch

# yapf: enable


class BC(RLAlgorithm):
    """Behavioral Cloning.

    Based on Model-Free Imitation Learning with Policy Optimization:
        https://arxiv.org/abs/1605.08478

    Args:
        env_spec (EnvSpec): Specification of environment.
        learner (garage.torch.Policy): Policy to train.
        batch_size (int): Size of optimization batch.
        source (Policy or Generator[TimeStepBatch]): Expert to clone. If a
            policy is passed, will set `.policy` to source and use the runner
            to sample from the policy.
        max_episode_length (int or None): Required if a policy is passed as
            source.
        policy_optimizer (torch.optim.Optimizer): Optimizer to be used to
            optimize the policy.
        policy_lr (float): Learning rate of the policy optimizer.
        loss (str): Which loss function to use. Must be either 'log_prob' or
            'mse'. If set to 'log_prob' (the default), `learner` must be a
            `garage.torch.StochasticPolicy`.
        minibatches_per_epoch (int): Number of minibatches per epoch.
        name (str): Name to use for logging.

    Raises:
        ValueError: If `source` is a `garage.Policy` and `max_episode_length`
            is not passed or `learner` is not a
            `garage.torch.StochasticPolicy` and loss is 'log_prob'.

    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        env_spec,
        learner,
        *,
        batch_size,
        source=None,
        max_episode_length=None,
        policy_optimizer=torch.optim.Adam,
        policy_lr=_Default(1e-3),
        loss='log_prob',
        minibatches_per_epoch=16,
        name='BC',
    ):
        self._source = source
        self.learner = learner
        self._optimizer = make_optimizer(policy_optimizer,
                                         module=self.learner,
                                         lr=policy_lr)
        if loss not in ('log_prob', 'mse'):
            raise ValueError('Loss should be either "log_prob" or "mse".')
        self._loss = loss
        self._minibatches_per_epoch = minibatches_per_epoch
        self._eval_env = None
        self._batch_size = batch_size
        self._name = name

        # Public fields for sampling.
        self.env_spec = env_spec
        self.policy = None
        self.max_episode_length = max_episode_length
        self.sampler_cls = None
        if isinstance(self._source, Policy):
            if max_episode_length is None:
                raise ValueError('max_episode_length must be passed if the '
                                 'source is a policy')
            self.policy = self._source
            self.sampler_cls = RaySampler
            self._source = source
        else:
            self._source = itertools.cycle(iter(source))

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): Experiment runner, for services such as
                snapshotting and sampler control.

        """
        if not self._eval_env:
            self._eval_env = runner.get_env_copy()
        for epoch in runner.step_epochs():
            if self._eval_env is not None:
                log_performance(epoch,
                                obtain_evaluation_episodes(
                                    self.learner, self._eval_env),
                                discount=1.0)
            losses = self._train_once(runner, epoch)
            with tabular.prefix(self._name + '/'):
                tabular.record('MeanLoss', np.mean(losses))
                tabular.record('StdLoss', np.std(losses))

    def _train_once(self, runner, epoch):
        """Obtain samplers and train for one epoch.

        Args:
            runner (LocalRunner): Experiment runner, which may be used to
                obtain samples.
            epoch (int): The current epoch.

        Returns:
            List[float]: Losses.

        """
        batch = self._obtain_samples(runner, epoch)
        indices = np.random.permutation(len(batch.actions))
        minibatches = np.array_split(indices, self._minibatches_per_epoch)
        losses = []
        for minibatch in minibatches:
            observations = np_to_torch(batch.observations[minibatch])
            actions = np_to_torch(batch.actions[minibatch])
            self._optimizer.zero_grad()
            loss = self._compute_loss(observations, actions)
            loss.backward()
            losses.append(loss.item())
            self._optimizer.step()
        return losses

    def _obtain_samples(self, runner, epoch):
        """Obtain samples from self._source.

        Args:
            runner (LocalRunner): Experiment runner, which may be used to
                obtain samples.
            epoch (int): The current epoch.

        Returns:
            TimeStepBatch: Batch of samples.

        """
        if isinstance(self._source, Policy):
            batch = EpisodeBatch.from_list(self.env_spec,
                                           runner.obtain_samples(epoch))
            log_performance(epoch, batch, 1.0, prefix='Expert')
            return batch
        else:
            batches = []
            while (sum(len(batch.actions)
                       for batch in batches) < self._batch_size):
                batches.append(next(self._source))
            return TimeStepBatch.concatenate(*batches)

    def _compute_loss(self, observations, expert_actions):
        """Compute loss of self._learner on the expert_actions.

        Args:
            observations (torch.Tensor): Observations used to select actions.
                Has shape :math:`(B, O^*)`, where :math:`B` is the batch
                dimension and :math:`O^*` are the observation dimensions.
            expert_actions (torch.Tensor): The actions of the expert.
                Has shape :math:`(B, A^*)`, where :math:`B` is the batch
                dimension and :math:`A^*` are the action dimensions.

        Returns:
            torch.Tensor: The loss through which gradient can be propagated
                back to the learner. Depends on self._loss.

        """
        learner_output = self.learner(observations)
        if self._loss == 'mse':
            if isinstance(learner_output, torch.Tensor):
                # We must have a deterministic policy as the learner.
                learner_actions = learner_output
            else:
                # We must have a StochasticPolicy as the learner.
                action_dist, _ = learner_output
                learner_actions = action_dist.rsample()
            return torch.mean((expert_actions - learner_actions)**2)
        else:
            assert self._loss == 'log_prob'
            # We already checked that we have a StochasticPolicy as the learner
            action_dist, _ = learner_output
            return -torch.mean(action_dist.log_prob(expert_actions))
