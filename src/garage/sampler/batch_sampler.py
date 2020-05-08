"""Class with batch-based sampling."""
import warnings

from garage.sampler import parallel_sampler
from garage.sampler.sampler_deprecated import BaseSampler
from garage.sampler.utils import truncate_paths


class BatchSampler(BaseSampler):
    """Class with batch-based sampling.

    Args:
        algo (garage.np.algos.RLAlgorithm): The algorithm.
        env (gym.Env): The environment.

    """

    def __init__(self, algo, env):
        super().__init__(algo, env)
        warnings.warn(
            DeprecationWarning(
                'BatchSampler is deprecated, and will be removed in the next '
                'release. Please use one of the samplers which implements '
                'garage.sampler.Sampler, such as LocalSampler.'))

    def start_worker(self):
        """Start workers."""
        parallel_sampler.populate_task(self.env,
                                       self.algo.policy,
                                       scope=self.algo.scope)

    def shutdown_worker(self):
        """Shutdown workers."""
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
        """Sample the policy for new trajectories.

        Args:
            itr (int): Number of iteration.
            batch_size (int): Number of environment steps in one batch.
            whole_paths (bool): Whether to use whole path or truncated.

        Returns:
            list[dict]: A list of paths.

        """
        if not batch_size:
            batch_size = self.algo.max_path_length

        cur_params = self.algo.policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )

        return paths if whole_paths else truncate_paths(paths, batch_size)
