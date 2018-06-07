import atexit
from multiprocessing import Process
import numpy as np
import platform
from queue import Empty, Queue
import tensorflow as tf
from threading import Thread

from rllab.sampler.utils import rollout

__all__ = ["Plotter"]


class Plotter(object):
    def __init__(self, env, policy, sess=None, graph=None):
        self.env = env
        self.policy = policy
        self.sess = tf.get_default_session() if sess is None else sess
        self.graph = tf.get_default_graph() if graph is None else graph
        self.worker_thread = Thread(target=self._start_worker, daemon=True)
        self.queue = Queue()

    def _start_worker(self):
        env = None
        policy = None
        max_length = None
        initial_rollout = True
        try:
            with self.sess.as_default(), self.sess.graph.as_default():
                while True:
                    msgs = {}
                    # If true, block and yield processor
                    if initial_rollout:
                        msg = self.queue.get()
                        msgs[msg[0]] = msg[1:]
                        # Only fetch the last message of each type
                        while not self.queue.empty():
                            msg = self.queue.get()
                            msgs[msg[0]] = msg[1:]
                    else:
                        # Only fetch the last message of each type
                        while not self.queue.empty():
                            msg = self.queue.get_nowait()
                            msgs[msg[0]] = msg[1:]

                    if 'stop' in msgs:
                        break
                    elif 'update' in msgs:
                        env, policy = msgs['update']
                    elif 'demo' in msgs:
                        param_values, max_length = msgs['demo']
                        policy.set_param_values(param_values)
                        initial_rollout = False
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

    def shutdown(self):
        if self.worker_thread.is_alive():
            self.queue.put(['stop'])
            self.queue.task_done()
            self.queue.join()
            self.worker_thread.join()

    def start(self):
        if not self.worker_thread.is_alive():
            tf.get_variable_scope().reuse_variables()
            self.worker_thread.start()
            self.queue.put(['update', self.env, self.policy])
            self.queue.task_done()
            atexit.register(self.shutdown)

    def update_plot(self, policy, max_length=np.inf):
        if self.worker_thread.is_alive():
            self.queue.put(['demo', policy.get_param_values(), max_length])
            self.queue.task_done()
