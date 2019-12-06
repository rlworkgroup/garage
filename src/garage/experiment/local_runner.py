"""Provides algorithms with access to most of garage's features."""
import copy
import os
import time
import types

from dowel import logger, tabular

from garage.experiment.snapshotter import Snapshotter


class LocalRunner:
    """Base class of local runner.

    Use Runner.setup(algo, env) to setup algorithm and environement for runner
    and Runner.train() to start training.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        max_cpus (int): The maximum number of parallel sampler workers.

    Note:
        For the use of any TensorFlow environments, policies and algorithms,
        please use LocalTFRunner().

    Examples:
        | # to train
        | runner = LocalRunner()
        | env = Env(...)
        | policy = Policy(...)
        | algo = Algo(
        |         env=env,
        |         policy=policy,
        |         ...)
        | runner.setup(algo, env)
        | runner.train(n_epochs=100, batch_size=4000)

        | # to resume immediately.
        | runner = LocalRunner()
        | runner.restore(resume_from_dir)
        | runner.resume()

        | # to resume with modified training arguments.
        | runner = LocalRunner()
        | runner.restore(resume_from_dir)
        | runner.resume(n_epochs=20)

    """

    def __init__(self, snapshot_config, max_cpus=1):
        self._snapshotter = Snapshotter(snapshot_config.snapshot_dir,
                                        snapshot_config.snapshot_mode,
                                        snapshot_config.snapshot_gap)

        if max_cpus > 1:
            from garage.sampler import singleton_pool
            singleton_pool.initialize(max_cpus)
        self.has_setup = False
        self.plot = False

        self._setup_args = None
        self.train_args = None

    def setup(self, algo, env, sampler_cls=None, sampler_args=None):
        """Set up runner for algorithm and environment.

        This method saves algo and env within runner and creates a sampler.

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
        self.algo = algo
        self.env = env
        self.policy = self.algo.policy

        if sampler_args is None:
            sampler_args = {}
        if sampler_cls is None:
            sampler_cls = algo.sampler_cls
        self.sampler = sampler_cls(algo, env, **sampler_args)

        self.has_setup = True

        self._setup_args = types.SimpleNamespace(sampler_cls=sampler_cls,
                                                 sampler_args=sampler_args)

    def _start_worker(self):
        """Start Plotter and Sampler workers."""
        self.sampler.start_worker()
        if self.plot:
            from garage.tf.plotter import Plotter
            self.plotter = Plotter(self.env, self.policy)
            self.plotter.start()

    def _shutdown_worker(self):
        """Shutdown Plotter and Sampler workers."""
        self.sampler.shutdown_worker()
        if self.plot:
            self.plotter.close()

    def obtain_samples(self, itr, batch_size=None):
        """Obtain one batch of samples.

        Args:
            itr(int): Index of iteration (epoch).
            batch_size(int): Number of steps in batch.
                This is a hint that the sampler may or may not respect.

        Returns:
            One batch of samples.

        """
        if self.train_args.n_epoch_cycles == 1:
            logger.log('Obtaining samples...')
        return self.sampler.obtain_samples(
            itr, (batch_size or self.train_args.batch_size))

    def save(self, epoch, paths=None):
        """Save snapshot of current batch.

        Args:
            itr(int): Index of iteration (epoch).
            paths(dict): Batch of samples after preprocessed. If None,
                no paths will be logged to the snapshot.

        """
        if not self.has_setup:
            raise Exception('Use setup() to setup runner before saving.')

        logger.log('Saving snapshot...')

        params = dict()
        # Save arguments
        params['setup_args'] = self._setup_args
        params['train_args'] = self.train_args

        # Save states
        params['env'] = self.env
        params['algo'] = self.algo
        if paths:
            params['paths'] = paths
        params['last_epoch'] = epoch
        self._snapshotter.save_itr_params(epoch, params)

        logger.log('Saved')

    def restore(self, from_dir, from_epoch='last'):
        """Restore experiment from snapshot.

        Args:
            from_dir (str): Directory of the pickle file
                to resume experiment from.
            from_epoch (str or int): The epoch to restore from.
                Can be 'first', 'last' or a number.
                Not applicable when snapshot_mode='last'.

        Returns:
            A SimpleNamespace for train()'s arguments.

        """
        saved = self._snapshotter.load(from_dir, from_epoch)

        self._setup_args = saved['setup_args']
        self.train_args = saved['train_args']

        self.setup(env=saved['env'],
                   algo=saved['algo'],
                   sampler_cls=self._setup_args.sampler_cls,
                   sampler_args=self._setup_args.sampler_args)

        n_epochs = self.train_args.n_epochs
        last_epoch = saved['last_epoch']
        n_epoch_cycles = self.train_args.n_epoch_cycles
        batch_size = self.train_args.batch_size
        store_paths = self.train_args.store_paths
        pause_for_plot = self.train_args.pause_for_plot

        fmt = '{:<20} {:<15}'
        logger.log('Restore from snapshot saved in %s' %
                   self._snapshotter.snapshot_dir)
        logger.log(fmt.format('Train Args', 'Value'))
        logger.log(fmt.format('n_epochs', n_epochs))
        logger.log(fmt.format('last_epoch', last_epoch))
        logger.log(fmt.format('n_epoch_cycles', n_epoch_cycles))
        logger.log(fmt.format('batch_size', batch_size))
        logger.log(fmt.format('store_paths', store_paths))
        logger.log(fmt.format('pause_for_plot', pause_for_plot))

        self.train_args.start_epoch = last_epoch + 1
        return copy.copy(self.train_args)

    def log_diagnostics(self, pause_for_plot=False):
        """Log diagnostics.

        Args:
            pause_for_plot(bool): Pause for plot.

        """
        logger.log('Time %.2f s' % (time.time() - self._start_time))
        logger.log('EpochTime %.2f s' % (time.time() - self._itr_start_time))
        logger.log(tabular)
        if self.plot:
            self.plotter.update_plot(self.policy, self.algo.max_path_length)
            if pause_for_plot:
                input('Plotting evaluation run: Press Enter to " "continue...')

    def train(self,
              n_epochs,
              batch_size,
              n_epoch_cycles=1,
              plot=False,
              store_paths=False,
              pause_for_plot=False):
        """Start training.

        Args:
            n_epochs(int): Number of epochs.
            batch_size(int): Number of environment steps in one batch.
            n_epoch_cycles(int): Number of batches of samples in each epoch.
                This is only useful for off-policy algorithm.
                For on-policy algorithm this value should always be 1.
            plot(bool): Visualize policy by doing rollout after each epoch.
            store_paths(bool): Save paths in snapshot.
            pause_for_plot(bool): Pause for plot.

        Returns:
            The average return in last epoch cycle.

        """
        if not self.has_setup:
            raise Exception('Use setup() to setup runner before training.')

        # Save arguments for restore
        self.train_args = types.SimpleNamespace(n_epochs=n_epochs,
                                                n_epoch_cycles=n_epoch_cycles,
                                                batch_size=batch_size,
                                                plot=plot,
                                                store_paths=store_paths,
                                                pause_for_plot=pause_for_plot,
                                                start_epoch=0)

        self.plot = plot

        return self.algo.train(self)

    def step_epochs(self):
        """Step through each epoch.

        This function returns a magic generator. When iterated through, this
        generator automatically performs services such as snapshotting and log
        management. It is used inside train() in each algorithm.

        The generator initializes two variables: `self.step_itr` and
        `self.step_path`. To use the generator, these two have to be
        updated manually in each epoch, as the example shows below.

        Yields:
            int: The next training epoch.

        Examples:
            for epoch in runner.step_epochs():
                runner.step_path = runner.obtain_samples(...)
                self.train_once(...)
                runner.step_itr += 1

        """
        try:
            self._start_worker()
            self._start_time = time.time()
            self.step_itr = (self.train_args.start_epoch *
                             self.train_args.n_epoch_cycles)
            self.step_path = None

            # Used by integration tests to ensure examples can run one epoch.
            n_epochs = int(
                os.environ.get('GARAGE_EXAMPLE_TEST_N_EPOCHS',
                               self.train_args.n_epochs))
            for epoch in range(self.train_args.start_epoch, n_epochs):
                self._itr_start_time = time.time()
                with logger.prefix('epoch #%d | ' % epoch):
                    yield epoch
                    save_path = (self.step_path
                                 if self.train_args.store_paths else None)
                    self.save(epoch, save_path)
                    self.log_diagnostics(self.train_args.pause_for_plot)
                    logger.dump_all(self.step_itr)
                    tabular.clear()
        finally:
            self._shutdown_worker()

    def resume(self,
               n_epochs=None,
               batch_size=None,
               n_epoch_cycles=None,
               plot=None,
               store_paths=None,
               pause_for_plot=None):
        """Resume from restored experiment.

        This method provides the same interface as train().

        If not specified, an argument will default to the
        saved arguments from the last call to train().

        Returns:
            The average return in last epoch cycle.

        """
        if self.train_args is None:
            raise Exception('You must call restore() before resume().')

        self.train_args.n_epochs = n_epochs or self.train_args.n_epochs
        self.train_args.batch_size = batch_size or self.train_args.batch_size
        self.train_args.n_epoch_cycles = (n_epoch_cycles
                                          or self.train_args.n_epoch_cycles)

        if plot is not None:
            self.train_args.plot = plot
        if store_paths is not None:
            self.train_args.store_paths = store_paths
        if pause_for_plot is not None:
            self.train_args.pause_for_plot = pause_for_plot

        return self.algo.train(self)
