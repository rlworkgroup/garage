import atexit
from collections import namedtuple
from enum import Enum
import platform
from queue import Queue
from threading import Thread

import numpy as np
import tensorflow as tf

from garage.sampler.utils import rollout as default_rollout

__all__ = ['Plotter']


class Op(Enum):
    STOP = 0
    UPDATE = 1
    DEMO = 2


Message = namedtuple('Message', ['op', 'args', 'kwargs'])


class Plotter:

    # Static variable used to disable the plotter
    enable = True
    # List containing all plotters instantiated in the process
    __plotters = []

    def __init__(self,
                 env,
                 policy,
                 sess=None,
                 graph=None,
                 rollout=default_rollout):
        Plotter.__plotters.append(self)
        self.env = env
        self.policy = policy
        self.sess = tf.compat.v1.get_default_session(
        ) if sess is None else sess
        self.graph = tf.compat.v1.get_default_graph(
        ) if graph is None else graph
        self.rollout = rollout
        self.worker_thread = Thread(target=self._start_worker, daemon=True)
        self.queue = Queue()

        # Needed in order to draw glfw window on the main thread
        if ('Darwin' in platform.platform()):
            self.rollout(
                env, policy, max_path_length=np.inf, animated=True, speedup=5)

    def _start_worker(self):
        env = None
        policy = None
        max_length = None
        initial_rollout = True
        try:
            with self.sess.as_default(), self.sess.graph.as_default():
                # Each iteration will process ALL messages currently in the
                # queue
                while True:
                    msgs = {}
                    # If true, block and yield processor
                    if initial_rollout:
                        msg = self.queue.get()
                        msgs[msg.op] = msg
                        # Only fetch the last message of each type
                        while not self.queue.empty():
                            msg = self.queue.get()
                            msgs[msg.op] = msg
                    else:
                        # Only fetch the last message of each type
                        while not self.queue.empty():
                            msg = self.queue.get_nowait()
                            msgs[msg.op] = msg

                    if Op.STOP in msgs:
                        self.queue.task_done()
                        break
                    if Op.UPDATE in msgs:
                        env, policy = msgs[Op.UPDATE].args
                        self.queue.task_done()
                    if Op.DEMO in msgs:
                        param_values, max_length = msgs[Op.DEMO].args
                        policy.set_param_values(param_values)
                        initial_rollout = False
                        self.rollout(
                            env,
                            policy,
                            max_path_length=max_length,
                            animated=True,
                            speedup=5)
                        self.queue.task_done()
                    else:
                        if max_length:
                            self.rollout(
                                env,
                                policy,
                                max_path_length=max_length,
                                animated=True,
                                speedup=5)
        except KeyboardInterrupt:
            pass

    def close(self):
        if self.worker_thread.is_alive():
            while not self.queue.empty():
                self.queue.get()
                self.queue.task_done()
            self.queue.put(Message(op=Op.STOP, args=None, kwargs=None))
            self.queue.join()
            self.worker_thread.join()

    @staticmethod
    def disable():
        """Disable all instances of the Plotter class."""
        Plotter.enable = False

    @staticmethod
    def get_plotters():
        return Plotter.__plotters

    def start(self):
        if not Plotter.enable:
            return
        if not self.worker_thread.is_alive():
            tf.compat.v1.get_variable_scope().reuse_variables()
            self.worker_thread.start()
            self.queue.put(
                Message(
                    op=Op.UPDATE, args=(self.env, self.policy), kwargs=None))
            atexit.register(self.close)

    def update_plot(self, policy, max_length=np.inf):
        if not Plotter.enable:
            return
        if self.worker_thread.is_alive():
            self.queue.put(
                Message(
                    op=Op.DEMO,
                    args=(policy.get_param_values(), max_length),
                    kwargs=None))
