"""A multiprocessing sampler which avoids waiting as much as possible."""
from collections import defaultdict
import itertools
import multiprocessing as mp
import queue

import click
import cloudpickle
import psutil
import setproctitle

from garage import EpisodeBatch
from garage.experiment.deterministic import get_seed
from garage.sampler.default_worker import DefaultWorker
from garage.sampler.sampler import Sampler
from garage.sampler.worker_factory import WorkerFactory


class MultiprocessingSampler(Sampler):
    """Sampler that uses multiprocessing to distribute workers.

    The sampler need to be created either from a worker factory or from
    parameters which can construct a worker factory. See the __init__ method
    of WorkerFactory for the detail of these parameters.

    Args:
        agents (Policy or List[Policy]): Agent(s) to use to sample episodes.
            If a list is passed in, it must have length exactly
            `worker_factory.n_workers`, and will be spread across the
            workers.
        envs (Environment or List[Environment]): Environment from which
            episodes are sampled. If a list is passed in, it must have length
            exactly `worker_factory.n_workers`, and will be spread across the
            workers.
        worker_factory (WorkerFactory): Pickleable factory for creating
            workers. Should be transmitted to other processes / nodes where
            work needs to be done, then workers should be constructed
            there. Either this param or params after this are required to
            construct a sampler.
        max_episode_length(int): Params used to construct a worker factory.
            The maximum length episodes which will be sampled.
        is_tf_worker (bool): Whether it is workers for TFTrainer.
        seed(int): The seed to use to initialize random number generators.
        n_workers(int): The number of workers to use.
        worker_class(type): Class of the workers. Instances should implement
            the Worker interface.
        worker_args (dict or None): Additional arguments that should be passed
            to the worker.

    """

    def __init__(
            self,
            agents,
            envs,
            *,  # After this require passing by keyword.
            worker_factory=None,
            max_episode_length=None,
            is_tf_worker=False,
            seed=get_seed(),
            n_workers=psutil.cpu_count(logical=False),
            worker_class=DefaultWorker,
            worker_args=None):
        # pylint: disable=super-init-not-called
        if worker_factory is None and max_episode_length is None:
            raise TypeError('Must construct a sampler from WorkerFactory or'
                            'parameters (at least max_episode_length)')
        if isinstance(worker_factory, WorkerFactory):
            self._factory = worker_factory
        else:
            self._factory = WorkerFactory(
                max_episode_length=max_episode_length,
                is_tf_worker=is_tf_worker,
                seed=seed,
                n_workers=n_workers,
                worker_class=worker_class,
                worker_args=worker_args)

        self._agents = self._factory.prepare_worker_messages(
            agents, cloudpickle.dumps)
        self._envs = self._factory.prepare_worker_messages(
            envs, cloudpickle.dumps)
        self._to_sampler = mp.Queue(2 * self._factory.n_workers)
        self._to_worker = [mp.Queue(1) for _ in range(self._factory.n_workers)]
        # If we crash from an exception, with full queues, we would rather not
        # hang forever, so we would like the process to close without flushing
        # the queues.
        # That's what cancel_join_thread does.
        for q in self._to_worker:
            q.cancel_join_thread()
        self._workers = [
            mp.Process(target=run_worker,
                       kwargs=dict(
                           factory=self._factory,
                           to_sampler=self._to_sampler,
                           to_worker=self._to_worker[worker_number],
                           worker_number=worker_number,
                           agent=self._agents[worker_number],
                           env=self._envs[worker_number],
                       ),
                       daemon=False)
            for worker_number in range(self._factory.n_workers)
        ]
        self._agent_version = 0
        for w in self._workers:
            w.start()
        self.total_env_steps = 0

    @classmethod
    def from_worker_factory(cls, worker_factory, agents, envs):
        """Construct this sampler.

        Args:
            worker_factory (WorkerFactory): Pickleable factory for creating
                workers. Should be transmitted to other processes / nodes where
                work needs to be done, then workers should be constructed
                there.
            agents (Policy or List[Policy]): Agent(s) to use to sample
                episodes. If a list is passed in, it must have length exactly
                `worker_factory.n_workers`, and will be spread across the
                workers.
            envs(Environment or List[Environment]): Environment from which
                episodes are sampled. If a list is passed in, it must have
                length exactly `worker_factory.n_workers`, and will be spread
                across the workers.

        Returns:
            Sampler: An instance of `cls`.

        """
        return cls(agents, envs, worker_factory=worker_factory)

    def _push_updates(self, updated_workers, agent_updates, env_updates):
        """Apply updates to the workers and (re)start them.

        Args:
            updated_workers (set[int]): Set of workers that don't need to be
                updated. Successfully updated workers will be added to this
                set.
            agent_updates (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_updates (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        """
        for worker_number, q in enumerate(self._to_worker):
            if worker_number in updated_workers:
                try:
                    q.put_nowait(('continue', ()))
                except queue.Full:
                    pass
            else:
                try:
                    q.put_nowait(('start', (agent_updates[worker_number],
                                            env_updates[worker_number],
                                            self._agent_version)))
                    updated_workers.add(worker_number)
                except queue.Full:
                    pass

    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        """Collect at least a given number transitions (timesteps).

        Args:
            itr(int): The current iteration number. Using this argument is
                deprecated.
            num_samples (int): Minimum number of transitions / timesteps to
                sample.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Returns:
            EpisodeBatch: The batch of collected episodes.

        Raises:
            AssertionError: On internal errors.

        """
        del itr
        batches = []
        completed_samples = 0
        self._agent_version += 1
        updated_workers = set()
        agent_ups = self._factory.prepare_worker_messages(
            agent_update, cloudpickle.dumps)
        env_ups = self._factory.prepare_worker_messages(
            env_update, cloudpickle.dumps)

        with click.progressbar(length=num_samples, label='Sampling') as pbar:
            while completed_samples < num_samples:
                self._push_updates(updated_workers, agent_ups, env_ups)
                for _ in range(self._factory.n_workers):
                    try:
                        tag, contents = self._to_sampler.get_nowait()
                        if tag == 'episode':
                            batch, version, worker_n = contents
                            del worker_n
                            if version == self._agent_version:
                                batches.append(batch)
                                num_returned_samples = batch.lengths.sum()
                                completed_samples += num_returned_samples
                                pbar.update(num_returned_samples)
                            else:
                                # Receiving episodes from previous iterations
                                # is normal.  Potentially, we could gather them
                                # here, if an off-policy method wants them.
                                pass
                        else:
                            raise AssertionError(
                                'Unknown tag {} with contents {}'.format(
                                    tag, contents))
                    except queue.Empty:
                        pass
            for q in self._to_worker:
                try:
                    q.put_nowait(('stop', ()))
                except queue.Full:
                    pass

        samples = EpisodeBatch.concatenate(*batches)
        self.total_env_steps += sum(samples.lengths)
        return samples

    def obtain_exact_episodes(self,
                              n_eps_per_worker,
                              agent_update,
                              env_update=None):
        """Sample an exact number of episodes per worker.

        Args:
            n_eps_per_worker (int): Exact number of episodes to gather for
                each worker.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Returns:
            EpisodeBatch: Batch of gathered episodes. Always in worker
                order. In other words, first all episodes from worker 0,
                then all episodes from worker 1, etc.

        Raises:
            AssertionError: On internal errors.

        """
        self._agent_version += 1
        updated_workers = set()
        agent_ups = self._factory.prepare_worker_messages(
            agent_update, cloudpickle.dumps)
        env_ups = self._factory.prepare_worker_messages(
            env_update, cloudpickle.dumps)
        episodes = defaultdict(list)

        with click.progressbar(length=self._factory.n_workers,
                               label='Sampling') as pbar:
            while any(
                    len(episodes[i]) < n_eps_per_worker
                    for i in range(self._factory.n_workers)):
                self._push_updates(updated_workers, agent_ups, env_ups)
                tag, contents = self._to_sampler.get()

                if tag == 'episode':
                    batch, version, worker_n = contents

                    if version == self._agent_version:
                        if len(episodes[worker_n]) < n_eps_per_worker:
                            episodes[worker_n].append(batch)

                        if len(episodes[worker_n]) == n_eps_per_worker:
                            pbar.update(1)
                            try:
                                self._to_worker[worker_n].put_nowait(
                                    ('stop', ()))
                            except queue.Full:
                                pass
                else:
                    raise AssertionError(
                        'Unknown tag {} with contents {}'.format(
                            tag, contents))

            for q in self._to_worker:
                try:
                    q.put_nowait(('stop', ()))
                except queue.Full:
                    pass

        ordered_episodes = list(
            itertools.chain(
                *[episodes[i] for i in range(self._factory.n_workers)]))
        samples = EpisodeBatch.concatenate(*ordered_episodes)
        self.total_env_steps += sum(samples.lengths)
        return samples

    def shutdown_worker(self):
        """Shutdown the workers."""
        for (q, w) in zip(self._to_worker, self._workers):
            # Loop until either the exit message is accepted or the process has
            # closed.  These might cause us to block, but ensures that the
            # workers are closed.
            while True:
                try:
                    # Set a timeout in case the child process crashed.
                    q.put(('exit', ()), timeout=1)
                    break
                except queue.Full:
                    # If the child process has crashed, we're done here.
                    # Otherwise it should eventually accept our message.
                    if not w.is_alive():
                        break
            # If this hangs forever, most likely a queue needs
            # cancel_join_thread called on it, or a subprocess has tripped the
            # "closing dowel with TensorboardOutput blocks forever bug."
            w.join()
        for q in self._to_worker:
            q.close()
        self._to_sampler.close()

    def __getstate__(self):
        """Get the pickle state.

        Returns:
            dict: The pickled state.

        """
        return dict(
            factory=self._factory,
            agents=[cloudpickle.loads(agent) for agent in self._agents],
            envs=[cloudpickle.loads(env) for env in self._envs])

    def __setstate__(self, state):
        """Unpickle the state.

        Args:
            state (dict): Unpickled state.

        """
        self.__init__(state['agents'],
                      state['envs'],
                      worker_factory=state['factory'])


def run_worker(factory, to_worker, to_sampler, worker_number, agent, env):
    """Run the streaming worker state machine.

    Starts in the "not streaming" state.
    Enters the "streaming" state when the "start" or "continue" message is
    received.
    While in the "streaming" state, it streams episodes back to the parent
    process.
    When it receives a "stop" message, or the queue back to the parent process
    is full, it enters the "not streaming" state.
    When it receives the "exit" message, it terminates.

    Critically, the worker never blocks on sending messages back to the
    sampler, to ensure it remains responsive to messages.

    Args:
        factory (WorkerFactory): Pickleable factory for creating workers.
            Should be transmitted to other processes / nodes where work needs
            to be done, then workers should be constructed there.
        to_worker (multiprocessing.Queue): Queue to send commands to the
            worker.
        to_sampler (multiprocessing.Queue): Queue to send episodes back to the
            sampler.
        worker_number (int): Number of this worker.
        agent (Policy): Agent to use to sample episodes.  If a list is passed
            in, it must have length exactly `worker_factory.n_workers`, and
            will be spread across the workers.
        env (Environment): Environment from which episodes are sampled. If a
            list is passed in, it must have length exactly
            `worker_factory.n_workers`, and will be spread across the workers.

    Raises:
        AssertionError: On internal errors.

    """
    # When a python process is closing, multiprocessing Queues attempt to flush
    # their contents to the underlying pipe.  If the pipe is full, they block
    # until someone reads from it.  In this case, the to_sampler pipe may be
    # full (or nearly full), and never read from, causing this process to hang
    # forever.  To avoid this, call cancel_join_thread, which will cause the
    # data to never be flushed to the pipe, allowing this process to end, and
    # the join on this process in the parent process to complete.
    # We call this immediately on process start in case this process crashes
    # (usually do to a bug or out-of-memory error in the underlying worker).
    to_sampler.cancel_join_thread()
    setproctitle.setproctitle('worker:' + setproctitle.getproctitle())

    inner_worker = factory(worker_number)
    inner_worker.update_agent(cloudpickle.loads(agent))
    inner_worker.update_env(cloudpickle.loads(env))

    version = 0
    streaming_samples = False

    while True:
        if streaming_samples:
            # We're streaming, so try to get a message without waiting. If we
            # can't, the message is "continue", without any contents.
            try:
                tag, contents = to_worker.get_nowait()
            except queue.Empty:
                tag = 'continue'
                contents = None
        else:
            # We're not streaming anymore, so wait for a message.
            tag, contents = to_worker.get()

        if tag == 'start':
            # Update env and policy.
            agent_update, env_update, version = contents
            inner_worker.update_agent(cloudpickle.loads(agent_update))
            inner_worker.update_env(cloudpickle.loads(env_update))
            streaming_samples = True
        elif tag == 'stop':
            streaming_samples = False
        elif tag == 'continue':
            batch = inner_worker.rollout()
            try:
                to_sampler.put_nowait(
                    ('episode', (batch, version, worker_number)))
            except queue.Full:
                # Either the sampler has fallen far behind the workers, or we
                # missed a "stop" message. Either way, stop streaming.
                # If the queue becomes empty again, the sampler will send a
                # continue (or some other) message.
                streaming_samples = False
        elif tag == 'exit':
            to_worker.close()
            to_sampler.close()
            inner_worker.shutdown()
            return
        else:
            raise AssertionError('Unknown tag {} with contents {}'.format(
                tag, contents))
