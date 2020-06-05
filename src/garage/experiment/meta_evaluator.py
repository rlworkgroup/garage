"""Evaluator which tests Meta-RL algorithms on test environments."""

from dowel import logger, tabular

from garage import log_multitask_performance, TrajectoryBatch
from garage.experiment.deterministic import get_seed
from garage.sampler import DefaultWorker
from garage.sampler import LocalSampler
from garage.sampler import WorkerFactory


class MetaEvaluator:
    """Evaluates Meta-RL algorithms on test environments.

    Args:
        test_task_sampler (garage.experiment.TaskSampler): Sampler for test
            tasks. To demonstrate the effectiveness of a meta-learning method,
            these should be different from the training tasks.
        max_path_length (int): Maximum path length used for evaluation
            trajectories.
        n_test_tasks (int or None): Number of test tasks to sample each time
            evaluation is performed. Note that tasks are sampled "without
            replacement". If None, is set to `test_task_sampler.n_tasks`.
        n_exploration_traj (int): Number of trajectories to gather from the
            exploration policy before requesting the meta algorithm to produce
            an adapted policy.
        n_test_rollouts (int): Number of rollouts to use for each adapted
            policy. The adapted policy should forget previous rollouts when
            `.reset()` is called.
        prefix (str): Prefix to use when logging. Defaults to MetaTest. For
            example, this results in logging the key 'MetaTest/SuccessRate'.
            If not set to `MetaTest`, it should probably be set to `MetaTrain`.
        test_task_names (list[str]): List of task names to test. Should be in
            an order consistent with the `task_id` env_info, if that is
            present.
        worker_class (type): Type of worker the Sampler should use.
        worker_args (dict or None): Additional arguments that should be
            passed to the worker.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 *,
                 test_task_sampler,
                 max_path_length,
                 n_exploration_traj=10,
                 n_test_tasks=None,
                 n_test_rollouts=1,
                 prefix='MetaTest',
                 test_task_names=None,
                 worker_class=DefaultWorker,
                 worker_args=None):
        self._test_task_sampler = test_task_sampler
        self._worker_class = worker_class
        if worker_args is None:
            self._worker_args = {}
        else:
            self._worker_args = worker_args
        if n_test_tasks is None:
            n_test_tasks = test_task_sampler.n_tasks
        self._n_test_tasks = n_test_tasks
        self._n_test_rollouts = n_test_rollouts
        self._n_exploration_traj = n_exploration_traj
        self._max_path_length = max_path_length
        self._eval_itr = 0
        self._prefix = prefix
        self._test_task_names = test_task_names
        self._test_sampler = None

    def evaluate(self, algo, test_rollouts_per_task=None):
        """Evaluate the Meta-RL algorithm on the test tasks.

        Args:
            algo (garage.np.algos.MetaRLAlgorithm): The algorithm to evaluate.
            test_rollouts_per_task (int or None): Number of rollouts per task.

        """
        if test_rollouts_per_task is None:
            test_rollouts_per_task = self._n_test_rollouts
        adapted_trajectories = []
        logger.log('Sampling for adapation and meta-testing...')
        if self._test_sampler is None:
            self._test_sampler = LocalSampler.from_worker_factory(
                WorkerFactory(seed=get_seed(),
                              max_path_length=self._max_path_length,
                              n_workers=1,
                              worker_class=self._worker_class,
                              worker_args=self._worker_args),
                agents=algo.get_exploration_policy(),
                envs=self._test_task_sampler.sample(1))
        for env_up in self._test_task_sampler.sample(self._n_test_tasks):
            policy = algo.get_exploration_policy()
            traj = TrajectoryBatch.concatenate(*[
                self._test_sampler.obtain_samples(self._eval_itr, 1, policy,
                                                  env_up)
                for _ in range(self._n_exploration_traj)
            ])
            adapted_policy = algo.adapt_policy(policy, traj)
            adapted_traj = self._test_sampler.obtain_samples(
                self._eval_itr, test_rollouts_per_task * self._max_path_length,
                adapted_policy)
            adapted_trajectories.append(adapted_traj)
        logger.log('Finished meta-testing...')

        if self._test_task_names is not None:
            name_map = dict(enumerate(self._test_task_names))
        else:
            name_map = None

        with tabular.prefix(self._prefix + '/' if self._prefix else ''):
            log_multitask_performance(
                self._eval_itr,
                TrajectoryBatch.concatenate(*adapted_trajectories),
                getattr(algo, 'discount', 1.0),
                name_map=name_map)
        self._eval_itr += 1
