"""Original parallel sampler pool backend."""
import pickle
import signal

from dowel import logger
import numpy as np

from garage.experiment import deterministic
from garage.sampler.stateful_pool import SharedGlobal
from garage.sampler.stateful_pool import singleton_pool
from garage.sampler.utils import rollout


def _worker_init(g, id):
    if singleton_pool.n_parallel > 1:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    g.worker_id = id


def initialize(n_parallel):
    """Initialize the worker pool."""
    # SIGINT is blocked for all processes created in parallel_sampler to avoid
    # the creation of sleeping and zombie processes.
    #
    # If the user interrupts run_experiment, there's a chance some processes
    # won't die due to a dead lock condition where one of the children in the
    # parallel sampler exits without releasing a lock once after it catches
    # SIGINT.
    #
    # Later the parent tries to acquire the same lock to proceed with his
    # cleanup, but it remains sleeping waiting for the lock to be released.
    # In the meantime, all the process in parallel sampler remain in the zombie
    # state since the parent cannot proceed with their clean up.
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGINT])
        singleton_pool.initialize(n_parallel)
        singleton_pool.run_each(_worker_init,
                                [(id, )
                                 for id in range(singleton_pool.n_parallel)])
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, [signal.SIGINT])


def _get_scoped_g(g, scope):
    if scope is None:
        return g
    if not hasattr(g, 'scopes'):
        g.scopes = dict()
    if scope not in g.scopes:
        g.scopes[scope] = SharedGlobal()
        g.scopes[scope].worker_id = g.worker_id
    return g.scopes[scope]


def _worker_populate_task(g, env, policy, scope=None):
    g = _get_scoped_g(g, scope)
    g.env = pickle.loads(env)
    g.policy = pickle.loads(policy)


def _worker_terminate_task(g, scope=None):
    g = _get_scoped_g(g, scope)
    if getattr(g, 'env', None):
        g.env.close()
        g.env = None
    if getattr(g, 'policy', None):
        g.policy.terminate()
        g.policy = None


def populate_task(env, policy, scope=None):
    """Set each worker's env and policy."""
    logger.log('Populating workers...')
    if singleton_pool.n_parallel > 1:
        singleton_pool.run_each(
            _worker_populate_task,
            [(pickle.dumps(env), pickle.dumps(policy), scope)] *
            singleton_pool.n_parallel)
    else:
        # avoid unnecessary copying
        g = _get_scoped_g(singleton_pool.G, scope)
        g.env = env
        g.policy = policy
    logger.log('Populated')


def terminate_task(scope=None):
    """Close each worker's env and terminate each policy."""
    singleton_pool.run_each(_worker_terminate_task,
                            [(scope, )] * singleton_pool.n_parallel)


def close():
    """Close the worker pool."""
    singleton_pool.close()


def _worker_set_seed(_, seed):
    logger.log('Setting seed to %d' % seed)
    deterministic.set_seed(seed)


def set_seed(seed):
    """Set the seed in each worker."""
    singleton_pool.run_each(_worker_set_seed,
                            [(seed + i, )
                             for i in range(singleton_pool.n_parallel)])


def _worker_set_policy_params(g, params, scope=None):
    g = _get_scoped_g(g, scope)
    g.policy.set_param_values(params)


def _worker_collect_one_path(g, max_path_length, scope=None):
    g = _get_scoped_g(g, scope)
    path = rollout(g.env, g.policy, max_path_length=max_path_length)
    return path, len(path['rewards'])


def sample_paths(policy_params,
                 max_samples,
                 max_path_length=np.inf,
                 scope=None):
    """Sample paths from each worker.

    :param policy_params: parameters for the policy. This will be updated on
     each worker process
    :param max_samples: desired maximum number of samples to be collected. The
     actual number of collected samples might be greater since all trajectories
     will be rolled out either until termination or until max_path_length is
     reached
    :param max_path_length: horizon / maximum length of a single trajectory
    :return: a list of collected paths
    """
    singleton_pool.run_each(_worker_set_policy_params,
                            [(policy_params, scope)] *
                            singleton_pool.n_parallel)
    return singleton_pool.run_collect(_worker_collect_one_path,
                                      threshold=max_samples,
                                      args=(max_path_length, scope),
                                      show_prog_bar=True)
