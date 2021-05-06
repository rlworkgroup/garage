"""Provides algorithms with access to most of garage's features."""
import copy
import os
import time

import cloudpickle
from dowel import logger, tabular

# This is avoiding a circular import
from garage.experiment.deterministic import get_seed, set_seed
from garage.experiment.experiment import dump_json
from garage.experiment.snapshotter import Snapshotter

tf = None


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


class Trainer:
    """Base class of trainer.

    Use trainer.setup(algo, env) to setup algorithm and environment for trainer
    and trainer.train() to start training.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by Trainer to create the snapshotter.
            If None, it will create one with default settings.

    Note:
        For the use of any TensorFlow environments, policies and algorithms,
        please use TFTrainer().

    Examples:
        | # to train
        | trainer = Trainer()
        | env = Env(...)
        | policy = Policy(...)
        | algo = Algo(
        |         env=env,
        |         policy=policy,
        |         ...)
        | trainer.setup(algo, env)
        | trainer.train(n_epochs=100, batch_size=4000)

        | # to resume immediately.
        | trainer = Trainer()
        | trainer.restore(resume_from_dir)
        | trainer.resume()

        | # to resume with modified training arguments.
        | trainer = Trainer()
        | trainer.restore(resume_from_dir)
        | trainer.resume(n_epochs=20)

    """

    def __init__(self, snapshot_config):
        self._snapshotter = Snapshotter(snapshot_config.snapshot_dir,
                                        snapshot_config.snapshot_mode,
                                        snapshot_config.snapshot_gap)

        self._has_setup = False
        self._plot = False

        self._seed = None
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

    def setup(self, algo, env):
        """Set up trainer for algorithm and environment.

        This method saves algo and env within trainer and creates a sampler.

        Note:
            After setup() is called all variables in session should have been
            initialized. setup() respects existing values in session so
            policy weights can be loaded before setup().

        Args:
            algo (RLAlgorithm): An algorithm instance. If this algo want to use
                samplers, it should have a `_sampler` field.
            env (Environment): An environment instance.

        """
        self._algo = algo
        self._env = env

        self._seed = get_seed()

        if hasattr(self._algo, '_sampler'):
            # pylint: disable=protected-access
            self._sampler = self._algo._sampler

        self._has_setup = True

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
            ValueError: If the trainer was initialized without a sampler, or
                batch_size wasn't provided here or to train.

        Returns:
            EpisodeBatch: Batch of episodes.

        """
        if self._sampler is None:
            raise ValueError('trainer was not initialized with `sampler`. '
                             'the algo should have a `_sampler` field when'
                             '`setup()` is called')
        if batch_size is None and self._train_args.batch_size is None:
            raise ValueError(
                'trainer was not initialized with `batch_size`. '
                'Either provide `batch_size` to trainer.train, '
                ' or pass `batch_size` to trainer.obtain_samples.')
        episodes = None
        if agent_update is None:
            policy = getattr(self._algo, 'exploration_policy', None)
            if policy is None:
                # This field should exist, since self.make_sampler would have
                # failed otherwise.
                policy = self._algo.policy
            agent_update = policy.get_param_values()
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
            ValueError: Raised if the trainer was initialized without a
                        sampler, or batch_size wasn't provided here
                        or to train.

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
            NotSetupError: if save() is called before the trainer is set up.

        """
        if not self._has_setup:
            raise NotSetupError('Use setup() to setup trainer before saving.')

        logger.log('Saving snapshot...')

        params = dict()
        # Save arguments
        params['seed'] = self._seed
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

        self._seed = saved['seed']
        self._train_args = saved['train_args']
        self._stats = saved['stats']

        set_seed(self._seed)

        self.setup(env=saved['env'], algo=saved['algo'])

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
            raise NotSetupError(
                'Use setup() to setup trainer before training.')

        # Save arguments for restore
        self._train_args = TrainArgs(n_epochs=n_epochs,
                                     batch_size=batch_size,
                                     plot=plot,
                                     store_episodes=store_episodes,
                                     pause_for_plot=pause_for_plot,
                                     start_epoch=0)

        self._plot = plot
        self._start_worker()

        log_dir = self._snapshotter.snapshot_dir
        summary_file = os.path.join(log_dir, 'experiment.json')
        dump_json(summary_file, self)

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
            for epoch in trainer.step_epochs():
                trainer.step_episode = trainer.obtain_samples(...)
                self.train_once(...)
                trainer.step_itr += 1

        """
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

    @total_env_steps.setter
    def total_env_steps(self, value):
        """Total environment steps collected.

        Args:
            value (int): Total environment steps collected.

        """
        self._stats.total_env_steps = value


class NotSetupError(Exception):
    """Raise when an experiment is about to run without setup."""


# pylint: disable=no-member
class TFTrainer(Trainer):
    """This class implements a trainer for TensorFlow algorithms.

    A trainer provides a default TensorFlow session using python context.
    This is useful for those experiment components (e.g. policy) that require a
    TensorFlow session during construction.

    Use trainer.setup(algo, env) to setup algorithm and environment for trainer
    and trainer.train() to start training.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by Trainer to create the snapshotter.
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
        with TFTrainer() as trainer:
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
            trainer.setup(algo, env)
            trainer.train(n_epochs=100, batch_size=4000)

        # to resume immediately.
        with TFTrainer() as trainer:
            trainer.restore(resume_from_dir)
            trainer.resume()

        # to resume with modified training arguments.
        with TFTrainer() as trainer:
            trainer.restore(resume_from_dir)
            trainer.resume(n_epochs=20)

    """

    def __init__(self, snapshot_config, sess=None):
        # pylint: disable=import-outside-toplevel
        import tensorflow
        # pylint: disable=global-statement
        global tf
        tf = tensorflow
        super().__init__(snapshot_config=snapshot_config)
        self.sess = sess or tf.compat.v1.Session()
        self.sess_entered = False

    def __enter__(self):
        """Set self.sess as the default session.

        Returns:
            TFTrainer: This trainer.

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

    def setup(self, algo, env):
        """Set up trainer and sessions for algorithm and environment.

        This method saves algo and env within trainer and creates a sampler,
        and initializes all uninitialized variables in session.

        Note:
            After setup() is called all variables in session should have been
            initialized. setup() respects existing values in session so
            policy weights can be loaded before setup().

        Args:
            algo (RLAlgorithm): An algorithm instance.
            env (Environment): An environment instance.

        """
        self.initialize_tf_vars()
        logger.log(self.sess.graph)
        super().setup(algo, env)

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
