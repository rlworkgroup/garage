"""Model-Agnostic Meta-Learning (MAML) algorithm implementation for RL."""
import collections
import copy

from dowel import tabular
import numpy as np
import torch

from garage import _Default, make_optimizer
from garage import log_multitask_performance
from garage import TrajectoryBatch
from garage.misc import tensor_utils
from garage.sampler import OnPolicyVectorizedSampler
from garage.tf.samplers import BatchSampler
from garage.torch.optimizers import ConjugateGradientOptimizer
from garage.torch.optimizers import DifferentiableSGD
import garage.torch.utils as tu


class MAML:
    """Model-Agnostic Meta-Learning (MAML).

    Args:
        inner_algo (garage.torch.algos.VPG): The inner algorithm used for
            computing loss.
        env (garage.envs.GarageEnv): A gym environment.
        policy (garage.torch.policies.Policy): Policy.
        meta_optimizer (Union[torch.optim.Optimizer, tuple]):
            Type of optimizer.
            This can be an optimizer type such as `torch.optim.Adam` or a tuple
            of type and dictionary, where dictionary contains arguments to
            initialize the optimizer e.g. `(torch.optim.Adam, {'lr' : 1e-3})`.
        meta_batch_size (int): Number of tasks sampled per batch.
        inner_lr (float): Adaptation learning rate.
        outer_lr (float): Meta policy learning rate.
        num_grad_updates (int): Number of adaptation gradient steps.
        meta_evaluator (garage.experiment.MetaEvaluator): A meta evaluator for
            meta-testing. If None, don't do meta-testing.
        evaluate_every_n_epochs (int): Do meta-testing every this epochs.

    """

    def __init__(self,
                 inner_algo,
                 env,
                 policy,
                 meta_optimizer,
                 meta_batch_size=40,
                 inner_lr=0.1,
                 outer_lr=1e-3,
                 num_grad_updates=1,
                 meta_evaluator=None,
                 evaluate_every_n_epochs=1):
        if policy.vectorized:
            self.sampler_cls = OnPolicyVectorizedSampler
        else:
            self.sampler_cls = BatchSampler

        self.max_path_length = inner_algo.max_path_length

        self._meta_evaluator = meta_evaluator
        self._policy = policy
        self._env = env
        self._value_function = copy.deepcopy(inner_algo._value_function)
        self._initial_vf_state = self._value_function.state_dict()
        self._num_grad_updates = num_grad_updates
        self._meta_batch_size = meta_batch_size
        self._inner_algo = inner_algo
        self._inner_optimizer = DifferentiableSGD(self._policy, lr=inner_lr)
        self._meta_optimizer = make_optimizer(meta_optimizer,
                                              module=policy,
                                              lr=_Default(outer_lr),
                                              eps=_Default(1e-5))
        self._evaluate_every_n_epochs = evaluate_every_n_epochs

    def train(self, runner):
        """Obtain samples and start training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in runner.step_epochs():
            all_samples, all_params = self._obtain_samples(runner)
            last_return = self.train_once(runner, all_samples, all_params)
            runner.step_itr += 1

        return last_return

    def train_once(self, runner, all_samples, all_params):
        """Train the algorithm once.

        Args:
            runner (garage.experiment.LocalRunner): The experiment runner.
            all_samples (list[list[MAMLTrajectoryBatch]]): A two
                dimensional list of MAMLTrajectoryBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).

        Returns:
            float: Average return.

        """
        itr = runner.step_itr
        old_theta = dict(self._policy.named_parameters())

        kl_before = self._compute_kl_constraint(all_samples,
                                                all_params,
                                                set_grad=False)

        meta_objective = self._compute_meta_loss(all_samples, all_params)

        self._meta_optimizer.zero_grad()
        meta_objective.backward()

        self._meta_optimize(all_samples, all_params)

        # Log
        loss_after = self._compute_meta_loss(all_samples,
                                             all_params,
                                             set_grad=False)
        kl_after = self._compute_kl_constraint(all_samples,
                                               all_params,
                                               set_grad=False)

        with torch.no_grad():
            policy_entropy = self._compute_policy_entropy(
                [task_samples[0] for task_samples in all_samples])
            average_return = self.log_performance(itr, all_samples,
                                                  meta_objective.item(),
                                                  loss_after.item(),
                                                  kl_before.item(),
                                                  kl_after.item(),
                                                  policy_entropy.mean().item())

        if self._meta_evaluator and itr % self._evaluate_every_n_epochs == 0:
            self._meta_evaluator.evaluate(self)

        tu.update_module_params(self._old_policy, old_theta)

        return average_return

    def _train_value_function(self, paths):
        """Train the value function.

        Args:
            paths (list[dict]): A list of collected paths.

        Returns:
            torch.Tensor: Calculated mean scalar value of value function loss
                (float).

        """
        # MAML resets a value function to its initial state before training.
        self._value_function.load_state_dict(self._initial_vf_state)

        obs = np.concatenate([path['observations'] for path in paths], axis=0)
        returns = np.concatenate([path['returns'] for path in paths])
        obs = torch.Tensor(obs)
        returns = torch.Tensor(returns)

        vf_loss = self._value_function.compute_loss(obs, returns)
        # pylint: disable=protected-access
        self._inner_algo._vf_optimizer.zero_grad()
        vf_loss.backward()
        # pylint: disable=protected-access
        self._inner_algo._vf_optimizer.step()

        return vf_loss

    def _obtain_samples(self, runner):
        """Obtain samples for each task before and after the fast-adaptation.

        Args:
            runner (LocalRunner): A local runner instance to obtain samples.

        Returns:
            tuple: Tuple of (all_samples, all_params).
                all_samples (list[MAMLTrajectoryBatch]): A list of size
                    [meta_batch_size * (num_grad_updates + 1)]
                all_params (list[dict]): A list of named parameter
                    dictionaries.

        """
        tasks = self._env.sample_tasks(self._meta_batch_size)
        all_samples = [[] for _ in range(len(tasks))]
        all_params = []
        theta = dict(self._policy.named_parameters())

        for i, task in enumerate(tasks):
            self._set_task(runner, task)

            for j in range(self._num_grad_updates + 1):
                paths = runner.obtain_samples(runner.step_itr)
                batch_samples = self._process_samples(paths)
                all_samples[i].append(batch_samples)

                # The last iteration does only sampling but no adapting
                if j < self._num_grad_updates:
                    # A grad need to be kept for the next grad update
                    # Except for the last grad update
                    require_grad = j < self._num_grad_updates - 1
                    self._adapt(batch_samples, set_grad=require_grad)

            all_params.append(dict(self._policy.named_parameters()))
            # Restore to pre-updated policy
            tu.update_module_params(self._policy, theta)

        return all_samples, all_params

    def _adapt(self, batch_samples, set_grad=True):
        """Performs one MAML inner step to update the policy.

        Args:
            batch_samples (MAMLTrajectoryBatch): Samples data for one
                task and one gradient step.
            set_grad (bool): if False, update policy parameters in-place.
                Else, allow taking gradient of functions of updated parameters
                with respect to pre-updated parameters.

        """
        # pylint: disable=protected-access
        loss = self._inner_algo._compute_loss(*batch_samples[1:])

        # Update policy parameters with one SGD step
        self._inner_optimizer.zero_grad()
        loss.backward(create_graph=set_grad)

        with torch.set_grad_enabled(set_grad):
            self._inner_optimizer.step()

    def _meta_optimize(self, all_samples, all_params):
        if isinstance(self._meta_optimizer, ConjugateGradientOptimizer):
            self._meta_optimizer.step(
                f_loss=lambda: self._compute_meta_loss(
                    all_samples, all_params, set_grad=False),
                f_constraint=lambda: self._compute_kl_constraint(
                    all_samples, all_params))
        else:
            self._meta_optimizer.step(lambda: self._compute_meta_loss(
                all_samples, all_params, set_grad=False))

    def _compute_meta_loss(self, all_samples, all_params, set_grad=True):
        """Compute loss to meta-optimize.

        Args:
            all_samples (list[list[MAMLTrajectoryBatch]]): A two
                dimensional list of MAMLTrajectoryBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).
            set_grad (bool): Whether to enable gradient calculation or not.

        Returns:
            torch.Tensor: Calculated mean value of loss.

        """
        theta = dict(self._policy.named_parameters())
        old_theta = dict(self._old_policy.named_parameters())

        losses = []
        for task_samples, task_params in zip(all_samples, all_params):
            for i in range(self._num_grad_updates):
                require_grad = i < self._num_grad_updates - 1 or set_grad
                self._adapt(task_samples[i], set_grad=require_grad)

            tu.update_module_params(self._old_policy, task_params)
            with torch.set_grad_enabled(set_grad):
                # pylint: disable=protected-access
                last_update = task_samples[-1]
                loss = self._inner_algo._compute_loss(*last_update[1:])
            losses.append(loss)

            tu.update_module_params(self._policy, theta)
            tu.update_module_params(self._old_policy, old_theta)

        return torch.stack(losses).mean()

    def _compute_kl_constraint(self, all_samples, all_params, set_grad=True):
        """Compute KL divergence.

        For each task, compute the KL divergence between the old policy
        distribution and current policy distribution.

        Args:
            all_samples (list[list[MAMLTrajectoryBatch]]): Two
                dimensional list of MAMLTrajectoryBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            all_params (list[dict]): A list of named parameter dictionaries.
                Each dictionary contains key value pair of names (str) and
                parameters (torch.Tensor).
            set_grad (bool): Whether to enable gradient calculation or not.

        Returns:
            torch.Tensor: Calculated mean value of KL divergence.

        """
        theta = dict(self._policy.named_parameters())
        old_theta = dict(self._old_policy.named_parameters())

        kls = []
        for task_samples, task_params in zip(all_samples, all_params):
            for i in range(self._num_grad_updates):
                require_grad = i < self._num_grad_updates - 1 or set_grad
                self._adapt(task_samples[i], set_grad=require_grad)

            tu.update_module_params(self._old_policy, task_params)
            with torch.set_grad_enabled(set_grad):
                # pylint: disable=protected-access
                kl = self._inner_algo._compute_kl_constraint(
                    task_samples[-1].observations)
            kls.append(kl)

            tu.update_module_params(self._policy, theta)
            tu.update_module_params(self._old_policy, old_theta)

        return torch.stack(kls).mean()

    def _compute_policy_entropy(self, task_samples):
        """Compute policy entropy.

        Args:
            task_samples (list[MAMLTrajectoryBatch]): Samples data for
                one task.

        Returns:
            torch.Tensor: Computed entropy value.

        """
        obs = torch.stack([samples.observations for samples in task_samples])
        # pylint: disable=protected-access
        entropies = self._inner_algo._compute_policy_entropy(obs)
        return entropies.mean()

    def _set_task(self, runner, task):
        # pylint: disable=protected-access, no-self-use
        for env in runner._sampler._vec_env.envs:
            env.set_task(task)

    @property
    def policy(self):
        """Current policy of the inner algorithm.

        Returns:
            garage.torch.policies.Policy: Current policy of the inner
                algorithm.

        """
        return self._policy

    @property
    def _old_policy(self):
        """Old policy of the inner algorithm.

        Returns:
            garage.torch.policies.Policy: Old policy of the inner algorithm.

        """
        # pylint: disable=protected-access
        return self._inner_algo._old_policy

    def _process_samples(self, paths):
        """Process sample data based on the collected paths.

        Args:
            paths (list[dict]): A list of collected paths.

        Returns:
            MAMLTrajectoryBatch: Processed samples data.

        """
        for path in paths:
            path['returns'] = tensor_utils.discount_cumsum(
                path['rewards'], self._inner_algo.discount).copy()

        self._train_value_function(paths)
        obs, actions, rewards, _, valids, baselines \
            = self._inner_algo.process_samples(paths)
        return MAMLTrajectoryBatch(paths, obs, actions, rewards, valids,
                                   baselines)

    def log_performance(self, itr, all_samples, loss_before, loss_after,
                        kl_before, kl, policy_entropy):
        """Evaluate performance of this batch.

        Args:
            itr (int): Iteration number.
            all_samples (list[list[MAMLTrajectoryBatch]]): Two
                dimensional list of MAMLTrajectoryBatch of size
                [meta_batch_size * (num_grad_updates + 1)]
            loss_before (float): Loss before optimization step.
            loss_after (float): Loss after optimization step.
            kl_before (float): KL divergence before optimization step.
            kl (float): KL divergence after optimization step.
            policy_entropy (float): Policy entropy.

        Returns:
            float: The average return in last epoch cycle.

        """
        tabular.record('Iteration', itr)

        name_map = None
        if hasattr(self._env, 'all_task_names'):
            names = self._env.all_task_names
            name_map = dict(zip(names, names))

        rtns = log_multitask_performance(
            itr,
            TrajectoryBatch.from_trajectory_list(
                env_spec=self._env.spec,
                paths=[
                    path for task_paths in all_samples
                    for path in task_paths[self._num_grad_updates].paths
                ]),
            discount=self._inner_algo.discount,
            name_map=name_map)

        with tabular.prefix(self._policy.name + '/'):
            tabular.record('LossBefore', loss_before)
            tabular.record('LossAfter', loss_after)
            tabular.record('dLoss', loss_before - loss_after)
            tabular.record('KLBefore', kl_before)
            tabular.record('KLAfter', kl)
            tabular.record('Entropy', policy_entropy)

        return np.mean(rtns)

    def get_exploration_policy(self):
        """Return a policy used before adaptation to a specific task.

        Each time it is retrieved, this policy should only be evaluated in one
        task.

        Returns:
            garage.Policy: The policy used to obtain samples that are later
                used for meta-RL adaptation.

        """
        return copy.deepcopy(self._policy)

    def adapt_policy(self, exploration_policy, exploration_trajectories):
        """Adapt the policy by one gradient steps for a task.

        Args:
            exploration_policy (garage.Policy): A policy which was returned
                from get_exploration_policy(), and which generated
                exploration_trajectories by interacting with an environment.
                The caller may not use this object after passing it into this
                method.
            exploration_trajectories (garage.TrajectoryBatch): Trajectories to
                adapt to, generated by exploration_policy exploring the
                environment.

        Returns:
            garage.Policy: A policy adapted to the task represented by the
                exploration_trajectories.

        """
        old_policy, self._policy = self._policy, exploration_policy
        self._inner_algo.policy = exploration_policy
        self._inner_optimizer.module = exploration_policy

        paths = exploration_trajectories.to_trajectory_list()
        batch_samples = self._process_samples(paths)

        self._adapt(batch_samples, set_grad=False)

        self._policy = old_policy
        self._inner_algo.policy = self._inner_optimizer.module = old_policy
        return exploration_policy


class MAMLTrajectoryBatch(
        collections.namedtuple('MAMLTrajectoryBatch', [
            'paths', 'observations', 'actions', 'rewards', 'valids',
            'baselines'
        ])):
    r"""A tuple representing a batch of whole trajectories in MAML.

    A :class:`MAMLTrajectoryBatch` represents a batch of whole trajectories
    produced from one environment.
    +-----------------------+-------------------------------------------------+
    | Symbol                | Description                                     |
    +=======================+=================================================+
    | :math:`N`             | Trajectory index dimension                      |
    +-----------------------+-------------------------------------------------+
    | :math:`T`             | Maximum length of a trajectory                  |
    +-----------------------+-------------------------------------------------+
    | :math:`S^*`           | Single-step shape of a time-series tensor       |
    +-----------------------+-------------------------------------------------+

    Attributes:
        paths (list[dict[str, np.ndarray or dict[str, np.ndarray]]]):
            Nonflatten original paths from sampler.
        observations (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet T, O^*)` containing the (possibly
            multi-dimensional) observations for all time steps in this batch.
            These must conform to :obj:`env_spec.observation_space`.
        actions (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet T, A^*)` containing the (possibly
            multi-dimensional) actions for all time steps in this batch. These
            must conform to :obj:`env_spec.action_space`.
        rewards (torch.Tensor): A torch tensor of shape
            :math:`(N \bullet T)` containing the rewards for all time
            steps in this batch.
        valids (numpy.ndarray): An integer numpy array of shape :math:`(N, )`
            containing the length of each trajectory in this batch. This may be
            used to reconstruct the individual trajectories.
        baselines (numpy.ndarray): An numpy array of shape
            :math:`(N \bullet T, )` containing the value function estimation
            at all time steps in this batch.

    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """
