"""Ray Sampler, for tensorflow algorithms.

Currently the same as garage.samplers.RaySampler but includes
support for Tensorflow sessions
"""
import ray
import tensorflow as tf

from garage.sampler import RaySampler, SamplerWorker


class RaySamplerTF(RaySampler):
    """Ray Sampler, for tensorflow algorithms.

    Currently the same as garage.samplers.RaySampler

    Args:
        - Same as garage.samplers.RaySampler
    """

    def __init__(self,
                 algo,
                 env,
                 seed,
                 should_render=False,
                 num_processors=None):
        super().__init__(algo,
                         env,
                         seed,
                         should_render=False,
                         num_processors=None,
                         sampler_worker_cls=SamplerWorkerTF)

    def shutdown_worker(self, local=False):
        """Shuts down the worker."""
        shutting_down = []
        for worker in self._all_workers.values():
            shutting_down.append(worker.shutdown.remote())
        ray.get(shutting_down)
        ray.shutdown()


class SamplerWorkerTF(SamplerWorker):
    """Sampler Worker for tensorflow on policy algorithms.

    - Same as garage.samplers.SamplerWorker, except it
    initializes a tensorflow session, because each worker
    is in a separate process.
    """

    def __init__(self,
                 worker_id,
                 env_pkl,
                 agent_pkl,
                 seed,
                 max_path_length,
                 should_render=False,
                 local=False):
        self._sess = tf.get_default_session()
        if not self._sess:
            # create a tf session for all
            # sampler worker processes in
            # order to execute the policy.
            self._sess = tf.Session()
            self._sess.__enter__()
        super().__init__(worker_id,
                         env_pkl,
                         agent_pkl,
                         seed,
                         max_path_length,
                         should_render=should_render)

    def shutdown(self):
        """Perform shutdown processes for TF."""
        if tf.get_default_session():
            self._sess.__exit__(None, None, None)
