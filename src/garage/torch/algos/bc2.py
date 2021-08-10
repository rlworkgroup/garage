"""Implementation of Behavioral Cloning in PyTorch."""
import click
from dowel import tabular
import numpy as np
import torch

from garage import (_Default, log_performance, make_optimizer,
                    obtain_evaluation_episodes)
from garage.np.algos.rl_algorithm import RLAlgorithm
from garage.torch import np_to_torch


class BC2(RLAlgorithm):
    """Behavioral Cloning using customizable optimizer.

    Based on Model-Free Imitation Learning with Policy Optimization:
        https://arxiv.org/abs/1605.08478

    Args:
        env_spec (garage.envs.EnvSpec): Specification of environment.
        learner (garage.torch.Policy): Policy to train.
        sampler (garage.sampler.Sampler): Sampler to use to acquire samples.
        batch_size (int): Size of optimization batch.
        max_eval_path_length (int or None): Required if a policy is passed as
            source.
        policy_optimizer (torch.optim.Optimizer): Optimizer to be used to
            optimize the policy.
        loss (str): Which loss function to use. Must be either 'log_prob' or
            'mse'. If set to 'log_prob' (the default), `learner` must be a
            `garage.torch.StochasticPolicy`.
        name (str): Name to use for logging.


    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        env_spec,
        learner,
        sampler,
        *,
        batch_size,
        sample_from_learner=False,
        eval_env=None,
        cycles_per_epoch=1,
        policy_optimizer=torch.optim.Adam,
        policy_lr=_Default(1e-3),
        n_eval_episodes=10,
        loss='log_prob',
        name='BC',
    ):
        self._env_spec = env_spec
        self.learner = learner
        self._sampler = sampler
        self._optimizer = make_optimizer(policy_optimizer,
                                         module=self.learner,
                                         lr=policy_lr)

        if loss not in ('log_prob', 'mse'):
            raise ValueError('Loss should be either "log_prob" or "mse".')
        self._loss = loss
        self._eval_env = eval_env
        self._batch_size = batch_size
        self._name = name
        self._cycles_per_epoch = cycles_per_epoch
        self._n_eval_episodes = n_eval_episodes
        self._sample_from_learner = sample_from_learner

        # For plotting
        self.policy = self.learner

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Trainer is passed to give algorithm
                the access to trainer.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        for epoch in trainer.step_epochs():
            losses = self._train_once(trainer, epoch)
            with tabular.prefix(self._name + '/'):
                tabular.record('MinLoss', np.min(losses))
                tabular.record('MaxLoss', np.max(losses))
                tabular.record('MeanLoss', np.mean(losses))
                tabular.record('StdLoss', np.std(losses))
            if self._eval_env is not None:
                eval_samples = obtain_evaluation_episodes(
                    self.learner,
                    self._eval_env,
                    max_episode_length=self._eval_env.spec.max_episode_length,
                    num_eps=self._n_eval_episodes)
                log_performance(epoch, eval_samples, discount=1.0)

    def _train_once(self, trainer, epoch):
        """Obtain samplers and train for one epoch.

        Args:
            trainer (Trainer): Experiment trainer, which may be used to
                obtain samples.
            epoch (int): The current epoch.

        Returns:
            List[float]: Losses.

        """
        with click.progressbar(range(self._cycles_per_epoch),
                               label='Cycles') as pbar:
            for _ in pbar:
                agent_update = None
                if self._sample_from_learner:
                    agent_update = self.learner
                batch = self._sampler.obtain_samples(epoch,
                                                     self._batch_size,
                                                     agent_update=agent_update)
                indices = np.random.permutation(len(batch.actions))
                n_minibatches = len(indices) / self._batch_size
                minibatches = np.array_split(indices, n_minibatches)
                losses = []
                for minibatch in minibatches:
                    observations = np_to_torch(batch.observations[minibatch])
                    actions = np_to_torch(batch.actions[minibatch])
                    self._optimizer.zero_grad()
                    loss = self._compute_loss(observations, actions)
                    loss.backward()
                    losses.append(loss.item())
                    self._optimizer.step()
        trainer.total_env_steps = self._sampler.total_env_steps
        return losses

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
