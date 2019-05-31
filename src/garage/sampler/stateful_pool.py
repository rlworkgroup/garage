import inspect
import multiprocessing as mp
import sys
import time
import traceback

from joblib.pool import MemmappingPool

from garage.misc.prog_bar_counter import ProgBarCounter


class SharedGlobal:
    pass


class StatefulPool:
    def __init__(self):
        self.n_parallel = 1
        self.pool = None
        self.queue = None
        self.worker_queue = None
        self.G = SharedGlobal()
        self.manager = None
        self.initialized = False

    def initialize(self, n_parallel):
        self.n_parallel = n_parallel
        if self.pool is not None:
            print('Warning: terminating existing pool')
            self.pool.terminate()
            self.pool = None
            self.queue.close()
            self.worker_queue.close()
            self.G = SharedGlobal()
        if n_parallel > 1:
            self.manager = mp.Manager()
            self.queue = mp.Queue()
            self.worker_queue = mp.Queue()
            self.pool = MemmappingPool(
                self.n_parallel,
                temp_folder='/tmp',
            )
        self.initialized = True

    def close(self):
        if self.manager:
            self.manager.shutdown()
        if self.pool:
            self.pool.close()

    def run_each(self, runner, args_list=None):
        """
        Run the method on each worker process, and collect the result of
        execution.

        The runner method will receive 'g' as its first argument, followed
        by the arguments in the args_list, if any
        :return:
        """
        assert not inspect.ismethod(runner), (
            'run_each() cannot run a class method. Please ensure that runner '
            'is a function with the prototype def foo(g, ...), where g is an '
            'object of type garage.sampler.stateful_pool.SharedGlobal')

        if args_list is None:
            args_list = [tuple()] * self.n_parallel
        assert len(args_list) == self.n_parallel
        if self.n_parallel > 1:
            results = self.pool.map_async(
                _worker_run_each, [(runner, args) for args in args_list])
            for i in range(self.n_parallel):
                self.worker_queue.get()
            for i in range(self.n_parallel):
                self.queue.put(None)
            return results.get()
        return [runner(self.G, *args_list[0])]

    def run_map(self, runner, args_list):
        assert not inspect.ismethod(runner), (
            'run_map() cannot run a class method. Please ensure that runner '
            "is a function with the prototype 'def foo(g, ...)', where g is "
            'an object of type garage.sampler.stateful_pool.SharedGlobal')

        if self.n_parallel > 1:
            return self.pool.map(_worker_run_map,
                                 [(runner, args) for args in args_list])
        else:
            ret = []
            for args in args_list:
                ret.append(runner(self.G, *args))
            return ret

    def run_imap_unordered(self, runner, args_list):
        assert not inspect.ismethod(runner), (
            'run_imap_unordered() cannot run a class method. Please ensure '
            "that runner is a function with the prototype 'def foo(g, ...)', "
            'where g is an object of type '
            'garage.sampler.stateful_pool.SharedGlobal')

        if self.n_parallel > 1:
            for x in self.pool.imap_unordered(
                    _worker_run_map, [(runner, args) for args in args_list]):
                yield x
        else:
            for args in args_list:
                yield runner(self.G, *args)

    def run_collect(self,
                    collect_once,
                    threshold,
                    args=None,
                    show_prog_bar=True):
        """
        Run the collector method using the worker pool. The collect_once method
        will receive 'g' as its first argument, followed by the provided args,
        if any. The method should return a pair of values. The first should be
        the object to be collected, and the second is the increment to be
        added.
        This will continue until the total increment reaches or exceeds the
        given threshold.

        Sample script:

        def collect_once(g):
            return 'a', 1

        stateful_pool.run_collect(collect_once, threshold=3)
        # should return ['a', 'a', 'a']

        :param collector:
        :param threshold:
        :return:
        """
        assert not inspect.ismethod(collect_once), (
            'run_collect() cannot run a class method. Please ensure that '
            "collect_once is a function with the prototype 'def foo(g, ...)', "
            'where g is an object of type '
            'garage.sampler.stateful_pool.SharedGlobal')

        if args is None:
            args = tuple()
        if self.pool:
            counter = self.manager.Value('i', 0)
            lock = self.manager.RLock()
            results = self.pool.map_async(_worker_run_collect, [
                (collect_once, counter, lock, threshold, args)
            ] * self.n_parallel)
            if show_prog_bar:
                pbar = ProgBarCounter(threshold)
            last_value = 0
            while True:
                time.sleep(0.1)
                with lock:
                    if counter.value >= threshold:
                        if show_prog_bar:
                            pbar.stop()
                        break
                    if show_prog_bar:
                        pbar.inc(counter.value - last_value)
                    last_value = counter.value
            return sum(results.get(), [])
        else:
            count = 0
            results = []
            if show_prog_bar:
                pbar = ProgBarCounter(threshold)
            while count < threshold:
                result, inc = collect_once(self.G, *args)
                results.append(result)
                count += inc
                if show_prog_bar:
                    pbar.inc(inc)
            if show_prog_bar:
                pbar.stop()
            return results
        return []


singleton_pool = StatefulPool()


def _worker_run_each(all_args):
    try:
        runner, args = all_args
        # signals to the master that this task is up and running
        singleton_pool.worker_queue.put(None)
        # wait for the master to signal continuation
        singleton_pool.queue.get()
        return runner(singleton_pool.G, *args)
    except Exception:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))


def _worker_run_collect(all_args):
    try:
        collect_once, counter, lock, threshold, args = all_args
        collected = []
        while True:
            with lock:
                if counter.value >= threshold:
                    return collected
            result, inc = collect_once(singleton_pool.G, *args)
            collected.append(result)
            with lock:
                counter.value += inc
                if counter.value >= threshold:
                    return collected
    except Exception:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))


def _worker_run_map(all_args):
    try:
        runner, args = all_args
        return runner(singleton_pool.G, *args)
    except Exception:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))
