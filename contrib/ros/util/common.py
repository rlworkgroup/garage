from functools import wraps
import threading
import time


def rate_limited(func):
    """Rate-limits the decorated function locally, for one process."""
    lock = threading.Lock()
    last_time_called = time.perf_counter()

    @wraps(func)
    def rate_limited_function(*args, **kwargs):
        lock.acquire()
        nonlocal last_time_called
        try:
            elapsed = time.perf_counter() - last_time_called
            left_to_wait = 1.0 / args[0].step_freq - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)

            return func(*args, **kwargs)
        finally:
            last_time_called = time.perf_counter()
            lock.release()

    return rate_limited_function
