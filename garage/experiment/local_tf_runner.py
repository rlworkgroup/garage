"""
The local runner for tensorflow algorithms.

A runner setup context for algorithms during initialization and
pipelines data between sampler and algorithm during training.
"""
import time

import tensorflow as tf

from garage.logger import logger
from garage.logger import snapshotter
from garage.logger import tabular

# Note: Optional module should be imported ad hoc to break circular dependency.


class LocalRunner:
    """This class implements a local runner for tensorflow algorithms.

    A local runner provides a default tensorflow session using python context.
    This is useful for those experiment components (e.g. policy) that require a
    tensorflow session during construction.

    Use Runner.setup(algo, env) to setup algorithm and environement for runner
    and Runner.train() to start training.

    Examples:
        with LocalRunner() as runner:
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

    """

    def __init__(self, sess=None, max_cpus=1):
        """Create a new local runner.

        Args:
            max_cpus: The maximum number of parallel sampler workers.
            sess: An optional tensorflow session.
                  A new session will be created immediately if not provided.

        Note:
            The local runner will set up a joblib task pool of size max_cpus
            possibly later used by BatchSampler. If BatchSampler is not used,
            the processes in the pool will remain dormant.

            This setup is required to use tensorflow in a multiprocess
            environment before a tensorflow session is created
            because tensorflow is not fork-safe.

            See https://github.com/tensorflow/tensorflow/issues/2448.

        """
        if max_cpus > 1:
            from garage.sampler import singleton_pool
            singleton_pool.initialize(max_cpus)
        self.sess = sess or tf.Session()
        self.sess_entered = False
        self.has_setup = False
        self.plot = False

    def __enter__(self):
        """Set self.sess as the default session.

        Returns:
            This local runner.

        """
        if tf.get_default_session() is not self.sess:
            self.sess.__enter__()
            self.sess_entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Leave session."""
        if tf.get_default_session() is self.sess and self.sess_entered:
            self.sess.__exit__(exc_type, exc_val, exc_tb)
            self.sess_entered = False

    def setup(self, algo, env, sampler_cls=None, sampler_args=None):
        """Set up runner for algorithm and environment.

        This method saves algo and env within runner and creates a sampler.

        Note:
            After setup() is called all variables in session should have been
            initialized. setup() respects existing values in session so
            policy weights can be loaded before setup().

        Args:
            algo: An algorithm instance.
            env: An environement instance.
            sampler_cls: A sampler class.
            sampler_args: Arguments to be passed to sampler constructor.

        """
        self.algo = algo
        self.env = env
        self.policy = self.algo.policy

        if sampler_args is None:
            sampler_args = {}

        if sampler_cls is None:
            from garage.tf.algos.batch_polopt import BatchPolopt
            if isinstance(algo, BatchPolopt):
                if self.policy.vectorized:
                    from garage.tf.samplers import OnPolicyVectorizedSampler
                    sampler_cls = OnPolicyVectorizedSampler
                else:
                    from garage.tf.samplers import BatchSampler
                    sampler_cls = BatchSampler
            else:
                from garage.tf.samplers import OffPolicyVectorizedSampler
                sampler_cls = OffPolicyVectorizedSampler

        self.sampler = sampler_cls(algo, env, **sampler_args)

        self.initialize_tf_vars()
        logger.log(self.sess.graph)
        self.has_setup = True

    def initialize_tf_vars(self):
        """Initialize all uninitialized variables in session."""
        with tf.name_scope('initialize_tf_vars'):
            uninited_set = [
                e.decode()
                for e in self.sess.run(tf.report_uninitialized_variables())
            ]
            self.sess.run(
                tf.variables_initializer([
                    v for v in tf.global_variables()
                    if v.name.split(':')[0] in uninited_set
                ]))

    def start_worker(self):
        """Start Plotter and Sampler workers."""
        self.sampler.start_worker()
        if self.plot:
            from garage.tf.plotter import Plotter
            self.plotter = Plotter(self.env, self.policy)
            self.plotter.start()

    def shutdown_worker(self):
        """Shutdown Plotter and Sampler workers."""
        self.sampler.shutdown_worker()
        if self.plot:
            self.plotter.close()

    def obtain_samples(self, itr, batch_size):
        """Obtain one batch of samples.

        Args:
            itr: Index of iteration (epoch).
            batch_size: Number of steps in batch.
                This is a hint that the sampler may or may not respect.

        Returns:
            One batch of samples.

        """
        if self.n_epoch_cycles == 1:
            logger.log('Obtaining samples...')
        return self.sampler.obtain_samples(itr, batch_size)

    def save_snapshot(self, itr, paths=None):
        """Save snapshot of current batch.

        Args:
            itr: Index of iteration (epoch).
            paths: Batch of samples after preprocessed.

        """
        assert self.has_setup

        logger.log('Saving snapshot...')
        params = self.algo.get_itr_snapshot(itr)
        params['env'] = self.env
        if paths:
            params['paths'] = paths
        snapshotter.save_itr_params(itr, params)
        logger.log('Saved')

    def log_diagnostics(self, pause_for_plot=False):
        """Log diagnostics.

        Args:
            pause_for_plot: Pause for plot.

        """
        logger.log('Time %.2f s' % (time.time() - self.start_time))
        logger.log('EpochTime %.2f s' % (time.time() - self.itr_start_time))
        logger.log(tabular)
        if self.plot:
            self.plotter.update_plot(self.policy, self.algo.max_path_length)
            if pause_for_plot:
                input('Plotting evaluation run: Press Enter to " "continue...')

    def train(self,
              n_epochs,
              n_epoch_cycles=1,
              batch_size=None,
              plot=False,
              store_paths=False,
              pause_for_plot=False):
        """Start training.

        Args:
            n_epochs: Number of epochs.
            n_epoch_cycles: Number of batches of samples in each epoch.
                This is only useful for off-policy algorithm.
                For on-policy algorithm this value should always be 1.
            batch_size: Number of steps in batch.
            plot: Visualize policy by doing rollout after each epoch.
            store_paths: Save paths in snapshot.
            pause_for_plot: Pause for plot.

        Returns:
            The average return in last epoch cycle.

        """
        assert self.has_setup, ('Use Runner.setup() to setup runner before '
                                'training.')
        if batch_size is None:
            from garage.tf.samplers import OffPolicyVectorizedSampler
            if isinstance(self.sampler, OffPolicyVectorizedSampler):
                batch_size = self.algo.max_path_length
            else:
                batch_size = 40 * self.algo.max_path_length

        self.n_epoch_cycles = n_epoch_cycles

        self.plot = plot
        self.start_worker()
        self.start_time = time.time()

        itr = 0
        last_return = None
        for epoch in range(n_epochs):
            self.itr_start_time = time.time()
            paths = None
            with logger.prefix('epoch #%d | ' % epoch):
                for cycle in range(n_epoch_cycles):
                    paths = self.obtain_samples(itr, batch_size)
                    paths = self.sampler.process_samples(itr, paths)
                    last_return = self.algo.train_once(itr, paths)
                    itr += 1
                self.save_snapshot(epoch, paths if store_paths else None)
                self.log_diagnostics(pause_for_plot)
                logger.dump_all(itr)
                tabular.clear()
        self.shutdown_worker()
        return last_return
