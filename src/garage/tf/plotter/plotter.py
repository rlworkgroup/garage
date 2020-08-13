"""Renders rollouts of the policy as it trains."""
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
    """Message types."""
    STOP = 0
    UPDATE = 1
    DEMO = 2


Message = namedtuple('Message', ['op', 'args', 'kwargs'])


class Plotter:
    """Visualizes episodes of the policy as it trains.

    Usually, this class is used by sending plot=True to LocalRunner.train().

    Args:
        env (gym.Env): Environment from which to visualize episodes. This will
        be used without copying in the current process but in a separate
        thread, so it should be given a unique copy (in particular, do not pass
        the training environment here, then try to pickle it, or you will
        occasionally get crashes).
        policy (garage.tf.Policy): Policy used to visualize episodes.
        sess (tf.Session): The TensorFlow session to use.
        graph (tf.Graph): The TensorFlow graph to use.
        rollout (callable): The rollout function to call.

    """

    # List containing all plotters instantiated in the process
    __plotters = []

    def __init__(self,
                 env,
                 policy,
                 sess=None,
                 graph=None,
                 rollout=default_rollout):
        Plotter.__plotters.append(self)
        self._env = env
        self.sess = tf.compat.v1.Session() if sess is None else sess
        self.graph = tf.compat.v1.get_default_graph(
        ) if graph is None else graph
        with self.sess.as_default(), self.graph.as_default():
            self._policy = policy.clone('plotter_policy')
        self.rollout = rollout
        self.worker_thread = Thread(target=self._start_worker, daemon=True)
        self.queue = Queue()

        # Needed in order to draw glfw window on the main thread
        if 'Darwin' in platform.platform():
            self.rollout(self._env,
                         self._policy,
                         max_episode_length=np.inf,
                         animated=True,
                         speedup=5)

    def _start_worker(self):
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
                        self._env, self._policy = msgs[Op.UPDATE].args
                        self.queue.task_done()
                    if Op.DEMO in msgs:
                        param_values, max_length = msgs[Op.DEMO].args
                        self._policy.set_param_values(param_values)
                        initial_rollout = False
                        self.rollout(self._env,
                                     self._policy,
                                     max_episode_length=max_length,
                                     animated=True,
                                     speedup=5)
                        self.queue.task_done()
                    else:
                        if max_length:
                            self.rollout(self._env,
                                         self._policy,
                                         max_episode_length=max_length,
                                         animated=True,
                                         speedup=5)
        except KeyboardInterrupt:
            pass

    def close(self):
        """Stop the Plotter's worker thread."""
        if self.worker_thread.is_alive():
            while not self.queue.empty():
                self.queue.get()
                self.queue.task_done()
            self.queue.put(Message(op=Op.STOP, args=None, kwargs=None))
            self.queue.join()
            self.worker_thread.join()

    @staticmethod
    def get_plotters():
        """Return all garage.tf.Plotter's.

        Returns:
            list[garage.tf.Plotter]: All the garage.tf.Plotter's

        """
        return Plotter.__plotters

    def start(self):
        """Start the Plotter's worker thread."""
        if not self.worker_thread.is_alive():
            tf.compat.v1.get_variable_scope().reuse_variables()
            self.worker_thread.start()
            self.queue.put(
                Message(op=Op.UPDATE,
                        args=(self._env, self._policy),
                        kwargs=None))
            atexit.register(self.close)

    def update_plot(self, policy, max_length=np.inf):
        """Update the policy being plotted.

        Args:
            policy (garage.tf.Policy): Policy to visualize.
            max_length (int or float): The maximum length to allow an episode
                to be. Defaults to infinity.

        """
        if self.worker_thread.is_alive():
            self.queue.put(
                Message(op=Op.DEMO,
                        args=(policy.get_param_values(), max_length),
                        kwargs=None))
