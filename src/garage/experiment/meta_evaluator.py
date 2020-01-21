"""Evaluator which tests Meta-RL algorithms on test environments."""
from dowel import tabular
import numpy as np

from garage.misc.tensor_utils import discount_cumsum
from garage.sampler import LocalSampler


class MetaEvaluator:
    """Evaluates Meta-RL algorithms on test environments.

    Args:
        runner (garage.experiment.LocalRunner): Runner.
        test_task_sampler (garage.experiment.TaskSampler): Sampler for test
            tasks. To demonstrate the effectiveness of a meta-learning method,
            these should be different from the training tasks.
        max_path_length (int): Maximum path length used for evaluation
            trajectories.
        n_test_tasks (int or None): Number of test tasks to sample each time
            evaluation is performed. Note that tasks are sampled "without
            replacement". If None, is set to `test_task_sampler.n_tasks`.
        prefix (str): Prefix to use when logging. Defaults to MetaTest. For
            example, this results in logging the key 'MetaTest/SuccessRate'.
            If not set to `MetaTest`, it should probably be set to `MetaTrain`.

    """

    def __init__(self,
                 runner,
                 *,
                 test_task_sampler,
                 max_path_length,
                 n_test_tasks=None,
                 prefix='MetaTest'):
        self._test_task_sampler = test_task_sampler
        if n_test_tasks is None:
            n_test_tasks = test_task_sampler.n_tasks
        self._n_test_tasks = n_test_tasks
        self._test_sampler = runner.make_sampler(
            LocalSampler, n_workers=1, max_path_length=max_path_length)
        self._eval_itr = 0
        self._prefix = prefix

    def evaluate(self, algo):
        """Evaluate the Meta-RL algorithm on the test tasks.

        Args:
            algo (garage.np.algos.MetaRLAlgorithm): The algorithm to evaluate.

        """
        policy = algo.policy
        adapted_trajectories = []
        for env_up in self._test_task_sampler.sample(self._n_test_tasks):
            traj = self._test_sampler.obtain_samples(self._eval_itr, 1, policy,
                                                     env_up)
            adapted_policy = algo.adapt(traj)
            adapted_traj = self._test_sampler.obtain_samples(
                self._eval_itr, 1, adapted_policy)
            adapted_trajectories.append(adapted_traj)
        self.log_evaluation(adapted_trajectories,
                            getattr(algo, 'discount', 1.0))

    def log_evaluation(self, trajectories, discount):
        """Log Meta-test evaluations to dowel.

        Args:
            trajectories (list[TrajectoryBatch]): The trajectories. We need to
                iterate through each trajectory to compute discounted returns,
                so there's no real point in batching them together.
            discount (float): Discount factor used for computing dicounted
                returns.

        """
        returns = []
        undiscounted_returns = []
        completion = []
        success = []
        for trajectory in trajectories:
            returns.append(discount_cumsum(trajectory.rewards, discount))
            undiscounted_returns.append(sum(trajectory.rewards))
            completion.append(float(trajectory.terminals.any()))
            success.append(
                float(
                    getattr(trajectory.env_infos, 'success',
                            np.asarray([])).any()))

        average_discounted_return = np.mean([r[0] for r in returns])

        tabular.record('{}/NumTasks'.format(self._prefix), self._n_test_tasks)
        tabular.record('{}/AverageDiscountedReturn'.format(self._prefix),
                       average_discounted_return)
        tabular.record('{}/AverageReturn'.format(self._prefix),
                       np.mean(undiscounted_returns))
        tabular.record('{}/StdReturn'.format(self._prefix),
                       np.std(undiscounted_returns))
        tabular.record('{}/MaxReturn'.format(self._prefix),
                       np.max(undiscounted_returns))
        tabular.record('{}/MinReturn'.format(self._prefix),
                       np.min(undiscounted_returns))
        tabular.record('{}/CompletionRate'.format(self._prefix),
                       np.mean(completion))
        tabular.record('{}/SuccessRate'.format(self._prefix), np.mean(success))
