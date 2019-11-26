"""Provides algorithms with access to most of garage's features."""
import copy
import os
import time

from dowel import logger, tabular

from garage.experiment.deterministic import get_seed, set_seed
from garage.experiment.snapshotter import Snapshotter


class ExperimentStats:
    # pylint: disable=too-few-public-methods
    """Statistics of a experiment.

    Args:
        total_epoch (int): Total epoches.
        total_itr (int): Total Iterations.
        total_env_steps (int): Total environment steps collected.
        last_path (list[dict]): Last sampled paths.

    """

    def __init__(self, total_epoch, total_itr, total_env_steps, last_path):
        self.total_epoch = total_epoch
        self.total_itr = total_itr
        self.total_env_steps = total_env_steps
        self.last_path = last_path


class SetupArgs:
    # pylint: disable=too-few-public-methods
    """Arguments to setup a runner.

    Args:
        sampler_cls (garage.sampler.Sampler): A sampler class.
        sampler_args (dict): Arguments to be passed to sampler constructor.
        seed (int): Random seed.

    """

    def __init__(self, sampler_cls, sampler_args, seed):
        self.sampler_cls = sampler_cls
        self.sampler_args = sampler_args
        self.seed = seed


class TrainArgs:
    # pylint: disable=too-few-public-methods
    """Arguments to call train() or resume().

    Args:
        n_epochs (int): Number of epochs.
        n_epoch_cycles (int): Number of batches of samples in each epoch.
            This is only useful for off-policy algorithm.
            For on-policy algorithm this value should always be 1.
        batch_size (int): Number of environment steps in one batch.
        plot (bool): Visualize policy by doing rollout after each epoch.
        store_paths (bool): Save paths in snapshot.
        pause_for_plot (bool): Pause for plot.
        start_epoch (int): The starting epoch. Used for resume().

    """

    def __init__(self, n_epochs, n_epoch_cycles, batch_size, plot, store_paths,
                 pause_for_plot, start_epoch):
        self.n_epochs = n_epochs
        self.n_epoch_cycles = n_epoch_cycles
        self.batch_size = batch_size
        self.plot = plot
        self.store_paths = store_paths
        self.pause_for_plot = pause_for_plot
        self.start_epoch = start_epoch


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
            # pylint: disable=import-outside-toplevel
            from garage.sampler import singleton_pool
            singleton_pool.initialize(max_cpus)
        self._has_setup = False
        self._plot = False

        self._setup_args = None
        self._train_args = None
        self._stats = ExperimentStats(total_itr=0,
                                      total_env_steps=0,
                                      total_epoch=0,
                                      last_path=None)

        self._algo = None
        self._env = None
        self._policy = None
        self._sampler = None
        self._plotter = None

        self._start_time = None
        self._itr_start_time = None
        self.step_itr = None
        self.step_path = None

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
        self._algo = algo
        self._env = env
        self._policy = self._algo.policy

        if sampler_args is None:
            sampler_args = {}
        if sampler_cls is None:
            sampler_cls = algo.sampler_cls
        self._sampler = sampler_cls(algo, env, **sampler_args)

        self._has_setup = True

        self._setup_args = SetupArgs(sampler_cls=sampler_cls,
                                     sampler_args=sampler_args,
                                     seed=get_seed())

    def _start_worker(self):
        """Start Plotter and Sampler workers."""
        self._sampler.start_worker()
        if self._plot:
            # pylint: disable=import-outside-toplevel
            from garage.tf.plotter import Plotter
            self._plotter = Plotter(self._env, self._policy)
            self._plotter.start()

    def _shutdown_worker(self):
        """Shutdown Plotter and Sampler workers."""
        self._sampler.shutdown_worker()
        if self._plot:
            self._plotter.close()

    def obtain_samples(self, itr, batch_size=None):
        """Obtain one batch of samples.

        Args:
            itr (int): Index of iteration (epoch).
            batch_size (int): Number of steps in batch.
                This is a hint that the sampler may or may not respect.

        Returns:
            list[dict]: One batch of samples.

        """
        if self._train_args.n_epoch_cycles == 1:
            logger.log('Obtaining samples...')

        paths = self._sampler.obtain_samples(
            itr, (batch_size or self._train_args.batch_size))

        self._stats.total_env_steps += sum([len(p['rewards']) for p in paths])

        return paths

    def save(self, epoch):
        """Save snapshot of current batch.

        Args:
            epoch (int): Epoch.

        Raises:
            NotSetupError: if save() is called before the runner is set up.

        """
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup runner before saving.')

        logger.log('Saving snapshot...')

        params = dict()
        # Save arguments
        params['setup_args'] = self._setup_args
        params['train_args'] = self._train_args
        params['stats'] = self._stats

        # Save states
        params['env'] = self._env
        params['algo'] = self._algo

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
            TrainArgs: Arguments for train().

        """
        saved = self._snapshotter.load(from_dir, from_epoch)

        self._setup_args = saved['setup_args']
        self._train_args = saved['train_args']
        self._stats = saved['stats']

        set_seed(self._setup_args.seed)

        self.setup(env=saved['env'],
                   algo=saved['algo'],
                   sampler_cls=self._setup_args.sampler_cls,
                   sampler_args=self._setup_args.sampler_args)

        n_epochs = self._train_args.n_epochs
        last_epoch = self._stats.total_epoch
        last_itr = self._stats.total_itr
        total_env_steps = self._stats.total_env_steps
        n_epoch_cycles = self._train_args.n_epoch_cycles
        batch_size = self._train_args.batch_size
        store_paths = self._train_args.store_paths
        pause_for_plot = self._train_args.pause_for_plot

        fmt = '{:<20} {:<15}'
        logger.log('Restore from snapshot saved in %s' %
                   self._snapshotter.snapshot_dir)
        logger.log(fmt.format('-- Train Args --', '-- Value --'))
        logger.log(fmt.format('n_epochs', n_epochs))
        logger.log(fmt.format('last_epoch', last_epoch))
        logger.log(fmt.format('n_epoch_cycles', n_epoch_cycles))
        logger.log(fmt.format('batch_size', batch_size))
        logger.log(fmt.format('store_paths', store_paths))
        logger.log(fmt.format('pause_for_plot', pause_for_plot))
        logger.log(fmt.format('-- Stats --', '-- Value --'))
        logger.log(fmt.format('last_itr', last_itr))
        logger.log(fmt.format('total_env_steps', total_env_steps))

        self._train_args.start_epoch = last_epoch + 1
        return copy.copy(self._train_args)

    def log_diagnostics(self, pause_for_plot=False):
        """Log diagnostics.

        Args:
            pause_for_plot (bool): Pause for plot.

        """
        logger.log('Time %.2f s' % (time.time() - self._start_time))
        logger.log('EpochTime %.2f s' % (time.time() - self._itr_start_time))
        logger.log(tabular)
        if self._plot:
            self._plotter.update_plot(self._policy, self._algo.max_path_length)
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
            n_epochs (int): Number of epochs.
            batch_size (int): Number of environment steps in one batch.
            n_epoch_cycles (int): Number of batches of samples in each epoch.
                This is only useful for off-policy algorithm.
                For on-policy algorithm this value should always be 1.
            plot (bool): Visualize policy by doing rollout after each epoch.
            store_paths (bool): Save paths in snapshot.
            pause_for_plot (bool): Pause for plot.

        Raises:
            NotSetupError: If train() is called before setup().

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup runner before training.')

        # Save arguments for restore
        self._train_args = TrainArgs(n_epochs=n_epochs,
                                     n_epoch_cycles=n_epoch_cycles,
                                     batch_size=batch_size,
                                     plot=plot,
                                     store_paths=store_paths,
                                     pause_for_plot=pause_for_plot,
                                     start_epoch=0)

        self._plot = plot

        return self._algo.train(self)

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
        self._start_worker()
        self._start_time = time.time()
        self.step_itr = self._stats.total_itr
        self.step_path = None

        # Used by integration tests to ensure examples can run one epoch.
        n_epochs = int(
            os.environ.get('GARAGE_EXAMPLE_TEST_N_EPOCHS',
                           self._train_args.n_epochs))

        for epoch in range(self._train_args.start_epoch, n_epochs):
            self._itr_start_time = time.time()
            with logger.prefix('epoch #%d | ' % epoch):
                yield epoch
                save_path = (self.step_path
                             if self._train_args.store_paths else None)

                self._stats.last_path = save_path
                self._stats.total_epoch = epoch
                self._stats.total_itr = self.step_itr

                self.save(epoch)
                self.log_diagnostics(self._train_args.pause_for_plot)
                logger.dump_all(self.step_itr)
                tabular.clear()

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

        Args:
            n_epochs (int): Number of epochs.
            batch_size (int): Number of environment steps in one batch.
            n_epoch_cycles (int): Number of batches of samples in each epoch.
                This is only useful for off-policy algorithm.
                For on-policy algorithm this value should always be 1.
            plot (bool): Visualize policy by doing rollout after each epoch.
            store_paths (bool): Save paths in snapshot.
            pause_for_plot (bool): Pause for plot.

        Raises:
            NotSetupError: If resume() is called before restore().

        Returns:
            float: The average return in last epoch cycle.

        """
        if self._train_args is None:
            raise NotSetupError('You must call restore() before resume().')

        self._train_args.n_epochs = n_epochs or self._train_args.n_epochs
        self._train_args.batch_size = batch_size or self._train_args.batch_size
        self._train_args.n_epoch_cycles = (n_epoch_cycles
                                           or self._train_args.n_epoch_cycles)

        if plot is not None:
            self._train_args.plot = plot
        if store_paths is not None:
            self._train_args.store_paths = store_paths
        if pause_for_plot is not None:
            self._train_args.pause_for_plot = pause_for_plot

        return self._algo.train(self)

    @property
    def total_env_steps(self):
        """Total environment steps collected.

        Returns:
            int: Total environment steps collected.

        """
        return self._stats.total_env_steps


class NotSetupError(Exception):
    """Raise when an experiment is about to run without setup."""
