import atexit
from multiprocessing import Process
from multiprocessing import Queue
import platform
from queue import Empty
from threading import Thread

import numpy as np

from garage.sampler.utils import rollout

__all__ = ['init_worker', 'init_plot', 'update_plot']

process = None
queue = None


def _worker_start():
    env = None
    policy = None
    max_length = None
    try:
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
                # env.start_viewer()
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


def _shutdown_worker():
    if process:
        queue.put(['stop'])
        queue.close()
        process.join()


def init_worker():
    global process, queue
    queue = Queue()
    process = Thread(target=_worker_start) if (
        'Darwin' in platform.platform()) else Process(target=_worker_start)
    process.daemon = True
    process.start()
    atexit.register(_shutdown_worker)


def init_plot(env, policy):
    global process, queue
    if not (process and queue):
        init_worker()

    # Needed in order to draw glfw window on the main thread
    if ('Darwin' in platform.platform()):
        rollout(env, policy, max_path_length=np.inf, animated=True, speedup=5)

    queue.put(['update', env, policy])


def update_plot(policy, max_length=np.inf):
    queue.put(['demo', policy.get_param_values(), max_length])
