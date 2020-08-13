"""Provides algorithms with access to most of garage's features."""
import copy
import os
import time

import cloudpickle
from dowel import logger, tabular
import psutil

# This is avoiding a circular import
from garage.experiment.deterministic import get_seed, set_seed
from garage.experiment.snapshotter import Snapshotter
from garage.sampler.default_worker import DefaultWorker
from garage.sampler.worker_factory import WorkerFactory


class ExperimentStats:
    # pylint: disable=too-few-public-methods
    """Statistics of a experiment.

    Args:
        total_epoch (int): Total epoches.
        total_itr (int): Total Iterations.
        total_env_steps (int): Total environment steps collected.
        last_episode (list[dict]): Last sampled episodes.

    """

    def __init__(self, total_epoch, total_itr, total_env_steps, last_episode):
        self.total_epoch = total_epoch
        self.total_itr = total_itr
        self.total_env_steps = total_env_steps
        self.last_episode = last_episode


class SetupArgs:
    # pylint: disable=too-few-public-methods
    """Arguments to setup a runner.

    Args:
        sampler_cls (Sampler): A sampler class.
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
        batch_size (int): Number of environment steps in one batch.
        plot (bool): Visualize an episode of the policy after after each epoch.
        store_episodes (bool): Save episodes in snapshot.
        pause_for_plot (bool): Pause for plot.
        start_epoch (int): The starting epoch. Used for resume().

    """

    def __init__(self, n_epochs, batch_size, plot, store_episodes,
                 pause_for_plot, start_epoch):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.plot = plot
        self.store_episodes = store_episodes
        self.pause_for_plot = pause_for_plot
        self.start_epoch = start_epoch


class LocalRunner:
    """Base class of local runner.

    Use Runner.setup(algo, env) to setup algorithm and environment for runner
    and Runner.train() to start training.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.

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

    def __init__(self, snapshot_config):
        self._snapshotter = Snapshotter(snapshot_config.snapshot_dir,
                                        snapshot_config.snapshot_mode,
                                        snapshot_config.snapshot_gap)

        self._has_setup = False
        self._plot = False

        self._setup_args = None
        self._train_args = None
        self._stats = ExperimentStats(total_itr=0,
                                      total_env_steps=0,
                                      total_epoch=0,
                                      last_episode=None)

        self._algo = None
        self._env = None
        self._sampler = None
        self._plotter = None

        self._start_time = None
        self._itr_start_time = None
        self.step_itr = None
        self.step_episode = None

        # only used for off-policy algorithms
        self.enable_logging = True

        self._n_workers = None
        self._worker_class = None
        self._worker_args = None

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
                the sampler. Epsiodes longer than this will be truncated.
            n_workers (int): The number of workers the sampler should use.
            worker_class (type): Type of worker the Sampler should use.
            sampler_args (dict or None): Additional arguments that should be
                passed to the sampler.
            worker_args (dict or None): Additional arguments that should be
                passed to the sampler.

        Raises:
            ValueError: If `max_episode_length` isn't passed and the algorithm
                doesn't contain a `max_episode_length` field, or if the
                algorithm doesn't have a policy field.

        Returns:
            sampler_cls: An instance of the sampler class.

        """
        policy = getattr(self._algo, 'exploration_policy', None)
        if policy is None:
            policy = getattr(self._algo, 'policy', None)
        if policy is None:
            raise ValueError('If the runner is used to construct a sampler, '
                             'the algorithm must have a `policy` or '
                             '`exploration_policy` field.')
        if max_episode_length is None:
            if hasattr(self._algo, 'max_episode_length'):
                max_episode_length = self._algo.max_episode_length
        if max_episode_length is None:
            raise ValueError('If `sampler_cls` is specified in runner.setup, '
                             'the algorithm must specify `max_episode_length`')
        if worker_class is None:
            worker_class = getattr(self._algo, 'worker_cls', DefaultWorker)
        if seed is None:
            seed = get_seed()
        if sampler_args is None:
            sampler_args = {}
        if worker_args is None:
            worker_args = {}

        return sampler_cls.from_worker_factory(WorkerFactory(
            seed=seed,
            max_episode_length=max_episode_length,
            n_workers=n_workers,
            worker_class=worker_class,
            worker_args=worker_args),
                                               agents=policy,
                                               envs=self._env)

    def setup(self,
              algo,
              env,
              sampler_cls=None,
              sampler_args=None,
              n_workers=psutil.cpu_count(logical=False),
              worker_class=DefaultWorker,
              worker_args=None):
        """Set up runner for algorithm and environment.

        This method saves algo and env within runner and creates a sampler.

        Note:
            After setup() is called all variables in session should have been
            initialized. setup() respects existing values in session so
            policy weights can be loaded before setup().

        Args:
            algo (RLAlgorithm): An algorithm instance.
            env (Environment): An environment instance.
            sampler_cls (type): A class which implements :class:`Sampler`.
            sampler_args (dict): Arguments to be passed to sampler constructor.
            n_workers (int): The number of workers the sampler should use.
            worker_class (type): Type of worker the sampler should use.
            worker_args (dict or None): Additional arguments that should be
                passed to the worker.

        Raises:
            ValueError: If sampler_cls is passed and the algorithm doesn't
                contain a `max_episode_length` field.

        """
        self._algo = algo
        self._env = env
        self._n_workers = n_workers
        self._worker_class = worker_class
        if sampler_args is None:
            sampler_args = {}
        if sampler_cls is None:
            sampler_cls = getattr(algo, 'sampler_cls', None)
        if worker_args is None:
            worker_args = {}

        self._worker_args = worker_args
        if sampler_cls is None:
            self._sampler = None
        else:
            self._sampler = self.make_sampler(sampler_cls,
                                              sampler_args=sampler_args,
                                              n_workers=n_workers,
                                              worker_class=worker_class,
                                              worker_args=worker_args)

        self._has_setup = True

        self._setup_args = SetupArgs(sampler_cls=sampler_cls,
                                     sampler_args=sampler_args,
                                     seed=get_seed())

    def _start_worker(self):
        """Start Plotter and Sampler workers."""
        if self._plot:
            # pylint: disable=import-outside-toplevel
            from garage.plotter import Plotter
            self._plotter = Plotter()
            self._plotter.init_plot(self.get_env_copy(), self._algo.policy)

    def _shutdown_worker(self):
        """Shutdown Plotter and Sampler workers."""
        if self._sampler is not None:
            self._sampler.shutdown_worker()
        if self._plot:
            self._plotter.close()

    def obtain_episodes(self,
                        itr,
                        batch_size=None,
                        agent_update=None,
                        env_update=None):
        """Obtain one batch of episodes.

        Args:
            itr (int): Index of iteration (epoch).
            batch_size (int): Number of steps in batch. This is a hint that the
                sampler may or may not respect.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before doing sampling episodes. If a list is
                passed in, it must have length exactly `factory.n_workers`, and
                will be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Raises:
            ValueError: If the runner was initialized without a sampler, or
                batch_size wasn't provided here or to train.

        Returns:
            EpisodeBatch: Batch of episodes.

        """
        if self._sampler is None:
            raise ValueError('Runner was not initialized with `sampler_cls`. '
                             'Either provide `sampler_cls` to runner.setup, '
                             ' or set `algo.sampler_cls`.')
        if batch_size is None and self._train_args.batch_size is None:
            raise ValueError('Runner was not initialized with `batch_size`. '
                             'Either provide `batch_size` to runner.train, '
                             ' or pass `batch_size` to runner.obtain_samples.')
        episodes = None
        if agent_update is None:
            agent_update = self._algo.policy.get_param_values()
        episodes = self._sampler.obtain_samples(
            itr, (batch_size or self._train_args.batch_size),
            agent_update=agent_update,
            env_update=env_update)
        self._stats.total_env_steps += sum(episodes.lengths)
        return episodes

    def obtain_samples(self,
                       itr,
                       batch_size=None,
                       agent_update=None,
                       env_update=None):
        """Obtain one batch of samples.

        Args:
            itr (int): Index of iteration (epoch).
            batch_size (int): Number of steps in batch.
                This is a hint that the sampler may or may not respect.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Raises:
            ValueError: Raised if the runner was initialized without a sampler,
                        or batch_size wasn't provided here or to train.

        Returns:
            list[dict]: One batch of samples.

        """
        eps = self.obtain_episodes(itr, batch_size, agent_update, env_update)
        return eps.to_list()

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
        params['n_workers'] = self._n_workers
        params['worker_class'] = self._worker_class
        params['worker_args'] = self._worker_args

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
                   sampler_args=self._setup_args.sampler_args,
                   n_workers=saved['n_workers'],
                   worker_class=saved['worker_class'],
                   worker_args=saved['worker_args'])

        n_epochs = self._train_args.n_epochs
        last_epoch = self._stats.total_epoch
        last_itr = self._stats.total_itr
        total_env_steps = self._stats.total_env_steps
        batch_size = self._train_args.batch_size
        store_episodes = self._train_args.store_episodes
        pause_for_plot = self._train_args.pause_for_plot

        fmt = '{:<20} {:<15}'
        logger.log('Restore from snapshot saved in %s' %
                   self._snapshotter.snapshot_dir)
        logger.log(fmt.format('-- Train Args --', '-- Value --'))
        logger.log(fmt.format('n_epochs', n_epochs))
        logger.log(fmt.format('last_epoch', last_epoch))
        logger.log(fmt.format('batch_size', batch_size))
        logger.log(fmt.format('store_episodes', store_episodes))
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
        tabular.record('TotalEnvSteps', self._stats.total_env_steps)
        logger.log(tabular)

        if self._plot:
            self._plotter.update_plot(self._algo.policy,
                                      self._algo.max_episode_length)
            if pause_for_plot:
                input('Plotting evaluation run: Press Enter to " "continue...')

    def train(self,
              n_epochs,
              batch_size=None,
              plot=False,
              store_episodes=False,
              pause_for_plot=False):
        """Start training.

        Args:
            n_epochs (int): Number of epochs.
            batch_size (int or None): Number of environment steps in one batch.
            plot (bool): Visualize an episode from the policy after each epoch.
            store_episodes (bool): Save episodes in snapshot.
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
                                     batch_size=batch_size,
                                     plot=plot,
                                     store_episodes=store_episodes,
                                     pause_for_plot=pause_for_plot,
                                     start_epoch=0)

        self._plot = plot

        average_return = self._algo.train(self)
        self._shutdown_worker()

        return average_return

    def step_epochs(self):
        """Step through each epoch.

        This function returns a magic generator. When iterated through, this
        generator automatically performs services such as snapshotting and log
        management. It is used inside train() in each algorithm.

        The generator initializes two variables: `self.step_itr` and
        `self.step_episode`. To use the generator, these two have to be
        updated manually in each epoch, as the example shows below.

        Yields:
            int: The next training epoch.

        Examples:
            for epoch in runner.step_epochs():
                runner.step_episode = runner.obtain_samples(...)
                self.train_once(...)
                runner.step_itr += 1

        """
        self._start_worker()
        self._start_time = time.time()
        self.step_itr = self._stats.total_itr
        self.step_episode = None

        # Used by integration tests to ensure examples can run one epoch.
        n_epochs = int(
            os.environ.get('GARAGE_EXAMPLE_TEST_N_EPOCHS',
                           self._train_args.n_epochs))

        logger.log('Obtaining samples...')

        for epoch in range(self._train_args.start_epoch, n_epochs):
            self._itr_start_time = time.time()
            with logger.prefix('epoch #%d | ' % epoch):
                yield epoch
                save_episode = (self.step_episode
                                if self._train_args.store_episodes else None)

                self._stats.last_episode = save_episode
                self._stats.total_epoch = epoch
                self._stats.total_itr = self.step_itr

                self.save(epoch)

                if self.enable_logging:
                    self.log_diagnostics(self._train_args.pause_for_plot)
                    logger.dump_all(self.step_itr)
                    tabular.clear()

    def resume(self,
               n_epochs=None,
               batch_size=None,
               plot=None,
               store_episodes=None,
               pause_for_plot=None):
        """Resume from restored experiment.

        This method provides the same interface as train().

        If not specified, an argument will default to the
        saved arguments from the last call to train().

        Args:
            n_epochs (int): Number of epochs.
            batch_size (int): Number of environment steps in one batch.
            plot (bool): Visualize an episode from the policy after each epoch.
            store_episodes (bool): Save episodes in snapshot.
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

        if plot is not None:
            self._train_args.plot = plot
        if store_episodes is not None:
            self._train_args.store_episodes = store_episodes
        if pause_for_plot is not None:
            self._train_args.pause_for_plot = pause_for_plot

        average_return = self._algo.train(self)
        self._shutdown_worker()

        return average_return

    def get_env_copy(self):
        """Get a copy of the environment.

        Returns:
            Environment: An environment instance.

        """
        if self._env:
            return cloudpickle.loads(cloudpickle.dumps(self._env))
        else:
            return None

    @property
    def total_env_steps(self):
        """Total environment steps collected.

        Returns:
            int: Total environment steps collected.

        """
        return self._stats.total_env_steps


class NotSetupError(Exception):
    """Raise when an experiment is about to run without setup."""
