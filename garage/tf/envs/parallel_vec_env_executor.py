import pickle as pickle
import uuid

import numpy as np

from garage.envs.util import flat_dim, flatten_n, unflatten_n
from garage.misc import logger
from garage.sampler import singleton_pool
from garage.tf.misc import tensor_utils


def worker_init_envs(G, alloc, scope, env):  # noqa
    logger.log("initializing environment on worker %d" % G.worker_id)
    if not hasattr(G, 'parallel_vec_envs'):
        G.parallel_vec_envs = dict()
        G.parallel_vec_env_template = dict()
    G.parallel_vec_envs[scope] = [(idx, pickle.loads(pickle.dumps(env)))
                                  for idx in alloc]
    G.parallel_vec_env_template[scope] = env


# For these two methods below, we pack the data into batch numpy arrays
# whenever possible, to reduce communication cost


def worker_run_reset(G, flags, scope):  # noqa
    if not hasattr(G, 'parallel_vec_envs'):
        logger.log("on worker %d" % G.worker_id)
        import traceback
        for line in traceback.format_stack():
            logger.log(line)
        # log the stacktrace at least
        logger.log("oops")
        for k, v in G.__dict__.items():
            logger.log(str(k) + " : " + str(v))
        assert hasattr(G, 'parallel_vec_envs')

    assert scope in G.parallel_vec_envs
    n = len(G.parallel_vec_envs[scope])
    env_template = G.parallel_vec_env_template[scope]
    obs_dim = flat_dim(env_template.observation_space)
    ret_arr = np.zeros((n, obs_dim))
    ids = []
    flat_obs = []
    reset_ids = []
    for itr_idx, (idx, env) in enumerate(G.parallel_vec_envs[scope]):
        flag = flags[idx]
        if flag:
            flat_obs.append(env.reset())
            reset_ids.append(itr_idx)
        ids.append(idx)
    if reset_ids:
        ret_arr[reset_ids] = flatten_n(env_template.observation_space,
                                       flat_obs)
    return ids, ret_arr


def worker_run_step(G, action_n, scope):  # noqa
    assert hasattr(G, 'parallel_vec_envs')
    assert scope in G.parallel_vec_envs
    env_template = G.parallel_vec_env_template[scope]
    ids = []
    step_results = []
    for (idx, env) in G.parallel_vec_envs[scope]:
        action = action_n[idx]
        ids.append(idx)
        step_results.append(tuple(env.step(action)))
    if not step_results:
        return None
    obs, rewards, dones, env_infos = list(map(list, list(zip(*step_results))))
    obs = flatten_n(env_template.observation_space, obs)
    rewards = np.asarray(rewards)
    dones = np.asarray(dones)
    env_infos = tensor_utils.stack_tensor_dict_list(env_infos)
    return ids, obs, rewards, dones, env_infos


def worker_collect_env_time(G):  # noqa
    return G.env_time


class ParallelVecEnvExecutor(object):
    def __init__(self, env, n, max_path_length, scope=None):
        if scope is None:
            # initialize random scope
            scope = str(uuid.uuid4())

        envs_per_worker = int(np.ceil(n * 1.0 / singleton_pool.n_parallel))
        alloc_env_ids = []
        rest_alloc = n
        start_id = 0
        for _ in range(singleton_pool.n_parallel):
            n_allocs = min(envs_per_worker, rest_alloc)
            alloc_env_ids.append(list(range(start_id, start_id + n_allocs)))
            start_id += n_allocs
            rest_alloc = max(0, rest_alloc - envs_per_worker)

        singleton_pool.run_each(
            worker_init_envs, [(alloc, scope, env) for alloc in alloc_env_ids])

        self._alloc_env_ids = alloc_env_ids
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self._num_envs = n
        self.scope = scope
        self.ts = np.zeros(n, dtype='int')
        self.max_path_length = max_path_length

    def step(self, action_n):
        results = singleton_pool.run_each(
            worker_run_step,
            [(action_n, self.scope) for _ in self._alloc_env_ids],
        )
        results = [x for x in results if x is not None]
        ids, obs, rewards, dones, env_infos = list(zip(*results))
        ids = np.concatenate(ids)
        obs = unflatten_n(self.observation_space, np.concatenate(obs))
        rewards = np.concatenate(rewards)
        dones = np.concatenate(dones)
        env_infos = tensor_utils.split_tensor_dict_list(
            tensor_utils.concat_tensor_dict_list(env_infos))
        if env_infos is None:
            env_infos = [dict() for _ in range(self.num_envs)]

        items = list(zip(ids, obs, rewards, dones, env_infos))
        items = sorted(items, key=lambda x: x[0])

        ids, obs, rewards, dones, env_infos = list(zip(*items))

        obs = list(obs)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)

        self.ts += 1
        dones[self.ts >= self.max_path_length] = True

        reset_obs = self._run_reset(dones)
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = reset_obs[i]
                self.ts[i] = 0
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(
            list(env_infos))

    def _run_reset(self, dones):
        dones = np.asarray(dones)
        results = singleton_pool.run_each(
            worker_run_reset,
            [(dones, self.scope) for _ in self._alloc_env_ids],
        )
        ids, flat_obs = list(map(np.concatenate, list(zip(*results))))
        zipped = list(zip(ids, flat_obs))
        sorted_obs = np.asarray(
            [x[1] for x in sorted(zipped, key=lambda x: x[0])])

        done_ids, = np.where(dones)
        done_flat_obs = sorted_obs[done_ids]
        done_unflat_obs = unflatten_n(self.observation_space, done_flat_obs)
        all_obs = [None] * self.num_envs
        done_cursor = 0
        for idx, done in enumerate(dones):
            if done:
                all_obs[idx] = done_unflat_obs[done_cursor]
                done_cursor += 1
        return all_obs

    def reset(self):
        dones = np.asarray([True] * self.num_envs)
        return self._run_reset(dones)

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def close(self):
        pass
