"""The local runner for TensorFlow algorithms.

A runner setup context for algorithms during initialization and
pipelines data between sampler and algorithm during training.
"""
from dowel import logger
import psutil

from garage.experiment import LocalRunner
from garage.sampler import DefaultWorker

# pylint: disable=no-name-in-module

tf = False
TFWorkerClassWrapper = False
try:
    import tensorflow as tf
    from garage.tf.samplers import TFWorkerClassWrapper  # noqa: E501; pylint: disable=ungrouped-imports
except ImportError:
    pass


class LocalTFRunner(LocalRunner):
    """This class implements a local runner for TensorFlow algorithms.

    A local runner provides a default TensorFlow session using python context.
    This is useful for those experiment components (e.g. policy) that require a
    TensorFlow session during construction.

    Use Runner.setup(algo, env) to setup algorithm and environment for runner
    and Runner.train() to start training.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        sess (tf.Session): An optional TensorFlow session.
              A new session will be created immediately if not provided.

    Note:
        When resume via command line, new snapshots will be
        saved into the SAME directory if not specified.

        When resume programmatically, snapshot directory should be
        specify manually or through @wrap_experiment interface.

    Examples:
        # to train
        with LocalTFRunner() as runner:
            env = gym.make('CartPole-v1')
            policy = CategoricalMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=(32, 32))
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                max_episode_length=100,
                discount=0.99,
                max_kl_step=0.01)
            runner.setup(algo, env)
            runner.train(n_epochs=100, batch_size=4000)

        # to resume immediately.
        with LocalTFRunner() as runner:
            runner.restore(resume_from_dir)
            runner.resume()

        # to resume with modified training arguments.
        with LocalTFRunner() as runner:
            runner.restore(resume_from_dir)
            runner.resume(n_epochs=20)

    """

    def __init__(self, snapshot_config, sess=None):
        super().__init__(snapshot_config=snapshot_config)
        self.sess = sess or tf.compat.v1.Session()
        self.sess_entered = False

    def __enter__(self):
        """Set self.sess as the default session.

        Returns:
            LocalTFRunner: This local runner.

        """
        if tf.compat.v1.get_default_session() is not self.sess:
            self.sess.__enter__()
            self.sess_entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave session.

        Args:
            exc_type (str): Type.
            exc_val (object): Value.
            exc_tb (object): Traceback.

        """
        if tf.compat.v1.get_default_session(
        ) is self.sess and self.sess_entered:
            self.sess.__exit__(exc_type, exc_val, exc_tb)
            self.sess_entered = False

    def make_sampler(self,
                     sampler_cls,
                     *,
                     seed=None,
                     n_workers=psutil.cpu_count(logical=False),
                     max_episode_length=None,
                     worker_class=None,
                     sampler_args=None,
                     worker_args=None):
        """Construct a Sampler from a Sampler class.

        Args:
            sampler_cls (type): The type of sampler to construct.
            seed (int): Seed to use in sampler workers.
            max_episode_length (int): Maximum episode length to be sampled by
                the sampler. Paths longer than this will be truncated.
            n_workers (int): The number of workers the sampler should use.
            worker_class (type): Type of worker the sampler should use.
            sampler_args (dict or None): Additional arguments that should be
                passed to the sampler.
            worker_args (dict or None): Additional arguments that should be
                passed to the worker.

        Returns:
            sampler_cls: An instance of the sampler class.

        """
        if worker_class is None:
            worker_class = getattr(self._algo, 'worker_cls', DefaultWorker)
        # pylint: disable=useless-super-delegation
        return super().make_sampler(
            sampler_cls,
            seed=seed,
            n_workers=n_workers,
            max_episode_length=max_episode_length,
            worker_class=TFWorkerClassWrapper(worker_class),
            sampler_args=sampler_args,
            worker_args=worker_args)

    def setup(self,
              algo,
              env,
              sampler_cls=None,
              sampler_args=None,
              n_workers=psutil.cpu_count(logical=False),
              worker_class=None,
              worker_args=None):
        """Set up runner and sessions for algorithm and environment.

        This method saves algo and env within runner and creates a sampler,
        and initializes all uninitialized variables in session.

        Note:
            After setup() is called all variables in session should have been
            initialized. setup() respects existing values in session so
            policy weights can be loaded before setup().

        Args:
            algo (RLAlgorithm): An algorithm instance.
            env (Environment): An environment instance.
            sampler_cls (type): A class which implements :class:`Sampler`
            sampler_args (dict): Arguments to be passed to sampler constructor.
            n_workers (int): The number of workers the sampler should use.
            worker_class (type): Type of worker the sampler should use.
            worker_args (dict or None): Additional arguments that should be
                passed to the worker.

        """
        self.initialize_tf_vars()
        logger.log(self.sess.graph)
        super().setup(algo, env, sampler_cls, sampler_args, n_workers,
                      worker_class, worker_args)

    def _start_worker(self):
        """Start Plotter and Sampler workers."""
        self._sampler.start_worker()
        if self._plot:
            # pylint: disable=import-outside-toplevel
            from garage.tf.plotter import Plotter
            self._plotter = Plotter(self.get_env_copy(),
                                    self._algo.policy,
                                    sess=tf.compat.v1.get_default_session())
            self._plotter.start()

    def initialize_tf_vars(self):
        """Initialize all uninitialized variables in session."""
        with tf.name_scope('initialize_tf_vars'):
            uninited_set = [
                e.decode() for e in self.sess.run(
                    tf.compat.v1.report_uninitialized_variables())
            ]
            self.sess.run(
                tf.compat.v1.variables_initializer([
                    v for v in tf.compat.v1.global_variables()
                    if v.name.split(':')[0] in uninited_set
                ]))


class __FakeLocalTFRunner:
    # noqa: E501; pylint: disable=missing-param-doc,too-few-public-methods,no-method-argument
    """Raises an ImportError for environments without TensorFlow."""

    def __init__(*args, **kwargs):
        raise ImportError(
            'LocalTFRunner requires TensorFlow. To use it, please install '
            'TensorFlow.')


if not tf:
    LocalTFRunner = __FakeLocalTFRunner  # noqa: F811
