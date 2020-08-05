"""Visualizes a policy running in an environment, in parallel with training."""
import atexit
from collections import namedtuple
from enum import Enum
from multiprocessing import JoinableQueue, Process
import platform
from threading import Thread

import numpy as np

from garage.sampler.utils import rollout

__all__ = ['Plotter']


class Op(Enum):
    """Messages for the Plotter state machine."""
    STOP = 0
    UPDATE = 1
    DEMO = 2


Message = namedtuple('Message', ['op', 'args', 'kwargs'])


class Plotter:
    """Visualizes a policy in an environment."""

    # Static variable used to disable the plotter
    enable = True
    # List containing all plotters instantiated in the process
    __plotters = []

    def __init__(self):
        Plotter.__plotters.append(self)
        self._process = None
        self._queue = None

    def _worker_start(self):
        env = None
        policy = None
        max_length = None
        initial_rollout = True
        try:
            # Each iteration will process ALL messages currently in the
            # queue
            while True:
                msgs = {}
                # If true, block and yield processor
                if initial_rollout:
                    msg = self._queue.get()
                    msgs[msg.op] = msg
                    # Only fetch the last message of each type
                    while not self._queue.empty():
                        msg = self._queue.get()
                        msgs[msg.op] = msg
                else:
                    # Only fetch the last message of each type
                    while not self._queue.empty():
                        msg = self._queue.get_nowait()
                        msgs[msg.op] = msg

                if Op.STOP in msgs:
                    break

                if Op.UPDATE in msgs:
                    env, policy = msgs[Op.UPDATE].args
                elif Op.DEMO in msgs:
                    param_values, max_length = msgs[Op.DEMO].args
                    policy.set_param_values(param_values)
                    initial_rollout = False
                    rollout(env,
                            policy,
                            max_episode_length=max_length,
                            animated=True,
                            speedup=5)
                else:
                    if max_length:
                        rollout(env,
                                policy,
                                max_episode_length=max_length,
                                animated=True,
                                speedup=5)
        except KeyboardInterrupt:
            pass

    def close(self):
        """Stop the plotter."""
        if not Plotter.enable:
            return
        if self._process and self._process.is_alive():
            while not self._queue.empty():
                self._queue.get()
                self._queue.task_done()
            self._queue.put(Message(op=Op.STOP, args=None, kwargs=None))
            self._queue.close()
            self._process.join()

    @staticmethod
    def disable():
        """Disable all instances of the Plotter class."""
        Plotter.enable = False

    @staticmethod
    def _get_plotters():
        """Get all instances of Plotter.

        Returns:
            List[Plotter]: All instances of Plotter.

        """
        return Plotter.__plotters

    def _init_worker(self):
        if not Plotter.enable:
            return
        self._queue = JoinableQueue()
        if 'Darwin' in platform.platform():
            self._process = Thread(target=self._worker_start)
        else:
            self._process = Process(target=self._worker_start)
        self._process.daemon = True
        self._process.start()
        atexit.register(self.close)

    def init_plot(self, env, policy):
        """Initialize the plotter.

        Args:
            env (GymEnv): Environment to visualize.
            policy (Policy): Policy to roll out in the
                visualization.

        """
        if not Plotter.enable:
            return
        if not (self._process and self._queue):
            self._init_worker()

        # Needed in order to draw glfw window on the main thread
        if 'Darwin' in platform.platform():
            rollout(env,
                    policy,
                    max_episode_length=np.inf,
                    animated=True,
                    speedup=5)

        self._queue.put(Message(op=Op.UPDATE, args=(env, policy), kwargs=None))

    def update_plot(self, policy, max_length=np.inf):
        """Update the plotter.

        Args:
            policy (garage.np.policies.Policy): New policy to roll out in the
                visualization.
            max_length (int): Maximum number of steps to roll out.

        """
        if not Plotter.enable:
            return
        self._queue.put(
            Message(op=Op.DEMO,
                    args=(policy.get_param_values(), max_length),
                    kwargs=None))
