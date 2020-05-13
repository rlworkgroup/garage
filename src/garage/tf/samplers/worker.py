"""Default TensorFlow sampler Worker."""
import numpy as np
import tensorflow as tf

from garage.sampler import Worker


class TFWorkerClassWrapper:
    """Acts like a Worker class, but is actually an object.

    When called, constructs the wrapped class and wraps it in a
    TFWorkerWrapper.

    Args:
        wrapped_class (type): The class to wrap. Should be a subclass of
            garage.sampler.Worker.

    """

    # pylint: disable=too-few-public-methods

    def __init__(self, wrapped_class):
        self._wrapped_class = wrapped_class

    def __call__(self, *args, **kwargs):
        """Construct the inner class and wrap it.

        Args:
            *args: Passed on to inner worker class.
            **kwargs: Passed on to inner worker class.

        Returns:
            TFWorkerWrapper: The wrapped worker.

        """
        wrapper = TFWorkerWrapper()
        # Need to construct the wrapped class after we've entered the Session.
        wrapper._inner_worker = self._wrapped_class(*args, **kwargs)
        return wrapper


class TFWorkerWrapper(Worker):
    """Wrapper around another workers that initializes a TensorFlow Session."""

    def __init__(self):
        # pylint: disable=super-init-not-called
        self._inner_worker = None
        self._sess = None
        self._sess_entered = None
        self.worker_init()

    def worker_init(self):
        """Initialize a worker."""
        self._sess = tf.compat.v1.get_default_session()
        if not self._sess:
            # create a tf session for all
            # sampler worker processes in
            # order to execute the policy.
            self._sess = tf.compat.v1.Session()
            self._sess_entered = True
            self._sess.__enter__()

    def shutdown(self):
        """Perform shutdown processes for TF."""
        self._inner_worker.shutdown()
        if tf.compat.v1.get_default_session() and self._sess_entered:
            self._sess_entered = False
            self._sess.__exit__(None, None, None)

    @property
    def agent(self):
        """Returns the worker's agent.

        Returns:
            garage.Policy: the worker's agent.

        """
        return self._inner_worker.agent

    @agent.setter
    def agent(self, agent):
        """Sets the worker's agent.

        Args:
            agent (garage.Policy): The agent.

        """
        self._inner_worker.agent = agent

    @property
    def env(self):
        """Returns the worker's environment.

        Returns:
            gym.Env: the worker's environment.

        """
        return self._inner_worker.env

    @env.setter
    def env(self, env):
        """Sets the worker's environment.

        Args:
            env (gym.Env): The environment.

        """
        self._inner_worker.env = env

    def update_agent(self, agent_update):
        """Update the worker's agent, using agent_update.

        Args:
            agent_update(object): An agent update. The exact type of this
                argument depends on the `Worker` implementation.

        """
        # flake8: noqa
        if not isinstance(
                agent_update,
            (dict, tuple, np.ndarray)) and (agent_update is not None) and (
                'default' not in agent_update.model.networks):
            obs_ph = tf.compat.v1.placeholder(tf.float32,
                                              shape=(None, None,
                                                     agent_update.obs_dim))
            agent_update.build(obs_ph)
        self._inner_worker.update_agent(agent_update)

    def update_env(self, env_update):
        """Update the worker's env, using env_update.

        Args:
            env_update(object): An environment update. The exact type of this
                argument depends on the `Worker` implementation.

        """
        self._inner_worker.update_env(env_update)

    def rollout(self):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: Batch of sampled trajectories. May be
                truncated if max_path_length is set.

        """
        return self._inner_worker.rollout()

    def start_rollout(self):
        """Begin a new rollout."""
        self._inner_worker.start_rollout()

    def step_rollout(self):
        """Take a single time-step in the current rollout.

        Returns:
            bool: True iff the path is done, either due to the environment
                indicating termination of due to reaching `max_path_length`.

        """
        return self._inner_worker.step_rollout()

    def collect_rollout(self):
        """Collect the current rollout, clearing the internal buffer.

        Returns:
            garage.TrajectoryBatch: Batch of sampled trajectories. May be
                truncated if the rollouts haven't completed yet.

        """
        return self._inner_worker.collect_rollout()
