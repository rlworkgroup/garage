"""Batch Sampler class."""
from garage.sampler import BaseSampler
from garage.sampler import parallel_sampler
from garage.sampler.utils import truncate_paths


class BatchSampler(BaseSampler):
    """Class with batch-based sampling."""

    def __init__(self, algo):
        """
        Init function.

        :type algo: BatchPolopt
        """
        self.algo = algo

    def start_worker(self):
        """Start worker function."""
        parallel_sampler.populate_task(
            self.algo.env, self.algo.policy, scope=self.algo.scope)

    def shutdown_worker(self):
        """Shutdown worker function."""
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr):
        """Obtain samples function."""
        cur_params = self.algo.policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.algo.batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if self.algo.whole_paths:
            return paths
        else:
            paths_truncated = truncate_paths(paths, self.algo.batch_size)
            return paths_truncated
