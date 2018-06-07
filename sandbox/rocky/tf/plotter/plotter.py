import atexit
from multiprocessing import Process
import numpy as np
import platform
from queue import Empty, Queue
import tensorflow as tf
from threading import Thread

from rllab.sampler.utils import rollout


class Plotter(object):
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.sess = tf.get_default_session()
        self.thread = None
        self.queue = None
        self._init_plot()

    def _init_plot(self):
        self._init_worker()
        self.queue.put(['update', self.env, self.policy])
        self.queue.task_done()

    def _init_worker(self):
        if not self.queue:
            self.queue = Queue()
            self.thread = Thread(
                target=self._start_worker, args=(
                    self.queue,
                    self.sess,
                ))
            self.thread.daemon = True
            self.thread.start()

    def _start_worker(self, queue, session):
        env = None
        policy = None
        max_length = None
        sess = session
        try:
            with sess.as_default():
                with sess.graph.as_default():
                    while True:
                        msgs = {}
                        # Only fetch the last message of each type
                        while True:
                            try:
                                msg = queue.get_nowait()
                                msgs[msg[0]] = msg[1:]
                            except Empty:
                                break
                        if 'stop' in msgs:
                            break
                        elif 'update' in msgs:
                            env, policy = msgs['update']
                        elif 'demo' in msgs:
                            param_values, max_length = msgs['demo']
                            policy.set_param_values(param_values)
                            rollout(
                                env,
                                policy,
                                max_path_length=max_length,
                                animated=True,
                                speedup=5)
                        else:
                            if max_length:
                                rollout(
                                    env,
                                    policy,
                                    max_path_length=max_length,
                                    animated=True,
                                    speedup=5)
        except KeyboardInterrupt:
            pass

    def _shutdown_worker(self):
        self.queue.put(['stop'])
        self.queue.task_done()
        self.queue.join()
        self.thread.join()

    def shutdown(self):
        self._shutdown_worker()

    def update_plot(self, policy, max_length=np.inf):
        self.queue.put(['demo', policy.get_param_values(), max_length])
        self.queue.task_done()
