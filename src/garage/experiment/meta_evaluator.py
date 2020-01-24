"""Evaluator which tests Meta-RL algorithms on test environments."""
from garage import log_performance, TrajectoryBatch
from garage.sampler import LocalSampler


class MetaEvaluator:
    """Evaluates Meta-RL algorithms on test environments.

    Args:
        runner (garage.experiment.LocalRunner): A runner capable of running
            policies from the (meta) algorithm. Can be the same runner used by
            the algorithm. Does not use runner.obtain_samples, and so does not
            affect TotalEnvSteps.
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
        prefix (str): Prefix to use when logging. Defaults to MetaTest. For
            example, this results in logging the key 'MetaTest/SuccessRate'.
            If not set to `MetaTest`, it should probably be set to `MetaTrain`.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self,
                 runner,
                 *,
                 test_task_sampler,
                 max_path_length,
                 n_test_tasks=None,
                 n_exploration_traj=1,
                 prefix='MetaTest'):
        self._test_task_sampler = test_task_sampler
        if n_test_tasks is None:
            n_test_tasks = test_task_sampler.n_tasks
        self._n_test_tasks = n_test_tasks
        self._n_exploration_traj = n_exploration_traj
        self._test_sampler = runner.make_sampler(
            LocalSampler, n_workers=1, max_path_length=max_path_length)
        self._eval_itr = 0
        self._prefix = prefix

    def evaluate(self, algo):
        """Evaluate the Meta-RL algorithm on the test tasks.

        Args:
            algo (garage.np.algos.MetaRLAlgorithm): The algorithm to evaluate.

        """
        adapted_trajectories = []
        for env_up in self._test_task_sampler.sample(self._n_test_tasks):
            policy = algo.get_exploration_policy()
            traj = TrajectoryBatch.concatenate(*[
                self._test_sampler.obtain_samples(self._eval_itr, 1, policy,
                                                  env_up)
                for _ in range(self._n_exploration_traj)
            ])
            adapted_policy = algo.adapt_policy(policy, traj)
            adapted_traj = self._test_sampler.obtain_samples(
                self._eval_itr, 1, adapted_policy)
            adapted_trajectories.append(adapted_traj)
        log_performance(self._eval_itr,
                        TrajectoryBatch.concatenate(*adapted_trajectories),
                        getattr(algo, 'discount', 1.0),
                        prefix=self._prefix)
        self._eval_itr += 1
