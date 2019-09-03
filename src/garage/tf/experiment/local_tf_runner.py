"""
The local runner for TensorFlow algorithms.

A runner setup context for algorithms during initialization and
pipelines data between sampler and algorithm during training.
"""
from dowel import logger
import tensorflow as tf

from garage.experiment import LocalRunner


class LocalTFRunner(LocalRunner):
    """This class implements a local runner for TensorFlow algorithms.

    A local runner provides a default TensorFlow session using python context.
    This is useful for those experiment components (e.g. policy) that require a
    TensorFlow session during construction.

    Use Runner.setup(algo, env) to setup algorithm and environement for runner
    and Runner.train() to start training.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        max_cpus (int): The maximum number of parallel sampler workers.
        sess (tf.Session): An optional TensorFlow session.
              A new session will be created immediately if not provided.

    Note:
        The local runner will set up a joblib task pool of size max_cpus
        possibly later used by BatchSampler. If BatchSampler is not used,
        the processes in the pool will remain dormant.

        This setup is required to use TensorFlow in a multiprocess
        environment before a TensorFlow session is created
        because TensorFlow is not fork-safe. See
        https://github.com/tensorflow/tensorflow/issues/2448.

        When resume via command line, new snapshots will be
        saved into the SAME directory if not specified.

        When resume programmatically, snapshot directory should be
        specify manually or through run_experiment() interface.

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
                max_path_length=100,
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

    def __init__(self, snapshot_config, sess=None, max_cpus=1):
        super().__init__(snapshot_config=snapshot_config, max_cpus=max_cpus)
        self.sess = sess or tf.compat.v1.Session()
        self.sess_entered = False

    def __enter__(self):
        """Set self.sess as the default session.

        Returns:
            This local runner.

        """
        if tf.compat.v1.get_default_session() is not self.sess:
            self.sess.__enter__()
            self.sess_entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave session."""
        if tf.compat.v1.get_default_session(
        ) is self.sess and self.sess_entered:
            self.sess.__exit__(exc_type, exc_val, exc_tb)
            self.sess_entered = False

    def setup(self, algo, env, sampler_cls=None, sampler_args=None):
        """Set up runner and sessions for algorithm and environment.

        This method saves algo and env within runner and creates a sampler,
        and initializes all uninitialized variables in session.

        Note:
            After setup() is called all variables in session should have been
            initialized. setup() respects existing values in session so
            policy weights can be loaded before setup().

        Args:
            algo (garage.np.algos.RLAlgorithm): An algorithm instance.
            env (garage.envs.GarageEnv): An environement instance.
            sampler_cls (garage.sampler.Sampler): A sampler class.
            sampler_args (dict): Arguments to be passed to sampler constructor.

        """
        self.initialize_tf_vars()
        logger.log(self.sess.graph)
        super().setup(algo, env, sampler_cls, sampler_args)

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
