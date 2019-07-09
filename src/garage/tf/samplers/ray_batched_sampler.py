"""Ray Sampler, for tensorflow algorithms.

Currently the same as garage.samplers.RaySampler but includes
support for Tensorflow sessions
"""
import tensorflow as tf

from garage.sampler import RaySampler, SamplerWorker


class RaySamplerTF(RaySampler):
    """Ray Sampler, for tensorflow algorithms.

    Currently the same as garage.samplers.RaySampler

    Args:
        - Same as garage.samplers.RaySampler
    """

    def __init__(self, algo, env, should_render=False, num_processors=None):
        super().__init__(
            algo,
            env,
            should_render=False,
            num_processors=None,
            sampler_worker_cls=SamplerWorkerTF)


class SamplerWorkerTF(SamplerWorker):
    """Sampler Worker for tensorflow on policy algorithms.

    - Same as garage.samplers.SamplerWorker, except it
    initializes a tensorflow session, because each worker
    is in a separate process.

    """

    def __init__(self,
                 worker_id,
                 env,
                 agent,
                 max_path_length,
                 should_render=False):
        self.sess = tf.InteractiveSession()
        super().__init__(
            worker_id, env, agent, max_path_length, should_render=False)
