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
        worker_factory(garage.sampler.WorkerFactory): Used for worker behavior.
        agents(list[garage.Policy]): Agents to distribute across workers.
        envs(list[gym.Env]): Environments to distribute across workers.

    """

    def __init__(self, worker_factory, agents, envs):
        super().__init__(worker_factory,
                         agents,
                         envs,
                         sampler_worker_cls=SamplerWorkerTF)

    def shutdown_worker(self):
        """Shuts down the worker."""
        shutting_down = []
        for worker in self._all_workers.values():
            shutting_down.append(worker.shutdown.remote())
        ray.get(shutting_down)
        ray.shutdown()


class SamplerWorkerTF(SamplerWorker):
    """Sampler Worker for tensorflow on policy algorithms.

    Args:
        worker_id(int): The id of the sampler_worker
        env(gym.Env): The gym env
        agent_pkl(bytes): The pickled agent
        worker_factory(WorkerFactory): Factory to construct this worker's
            behavior.

    """

    def __init__(self, worker_id, env, agent_pkl, worker_factory):
        self._sess = tf.get_default_session()
        if not self._sess:
            # create a tf session for all
            # sampler worker processes in
            # order to execute the policy.
            self._sess = tf.Session()
            self._sess.__enter__()
        super().__init__(worker_id, env, agent_pkl, worker_factory)

    def shutdown(self):
        """Perform shutdown processes for TF."""
        if tf.get_default_session():
            self._sess.__exit__(None, None, None)
