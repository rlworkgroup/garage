import atexit
from collections import namedtuple
from enum import Enum
from queue import Queue
from threading import Thread

import numpy as np
import tensorflow as tf

from garage.sampler.utils import rollout as default_rollout

__all__ = ["Plotter"]


class Op(Enum):
    STOP = 0
    UPDATE = 1
    DEMO = 2


Message = namedtuple("Message", ["op", "args", "kwargs"])


class Plotter(object):

    # Static variable used along with function run_experiment to enable or
    # disable the plotter
    enable = None

    def __init__(self,
                 env,
                 policy,
                 sess=None,
                 graph=None,
                 rollout=default_rollout):
        self.env = env
        self.policy = policy
        self.sess = tf.get_default_session() if sess is None else sess
        self.graph = tf.get_default_graph() if graph is None else graph
        self.rollout = rollout
        self.worker_thread = Thread(target=self._start_worker, daemon=True)
        self.queue = Queue()

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
                        break
                    if Op.UPDATE in msgs:
                        env, policy = msgs[Op.UPDATE].args
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

    def shutdown(self):
        if self.worker_thread.is_alive():
            self.queue.put(Message(op=Op.STOP, args=None, kwargs=None))
            self.queue.task_done()
            self.queue.join()
            self.worker_thread.join()

    @staticmethod
    def set_enable(enable):
        """Set the plot enabling according to the run_experiment function.
        """
        Plotter.enable = enable

    @staticmethod
    def status(plot_algo_status):
        """Use this method as a guard for plot enabling.

        This guard method considers the plot enabling defined in the
        function run_experiment.

        Parameters
        ----------
        plot_algo_status : boolean
            The plot enabling defined in the algorithm class.

        Returns
        -------
            The plot enabling of the algorithm if the run_experiment was not
            called, and the plot enabling of the run_experiment otherwise.

        """
        if Plotter.enable is None:
            return plot_algo_status
        else:
            return Plotter.enable

    def start(self):
        if Plotter.enable is None or Plotter.enable:
            if not self.worker_thread.is_alive():
                tf.get_variable_scope().reuse_variables()
                self.worker_thread.start()
                self.queue.put(
                    Message(
                        op=Op.UPDATE,
                        args=(self.env, self.policy),
                        kwargs=None))
                self.queue.task_done()
                atexit.register(self.shutdown)

    def update_plot(self, policy, max_length=np.inf):
        if Plotter.enable is None or Plotter.enable:
            if self.worker_thread.is_alive():
                self.queue.put(
                    Message(
                        op=Op.DEMO,
                        args=(policy.get_param_values(), max_length),
                        kwargs=None))
                self.queue.task_done()
