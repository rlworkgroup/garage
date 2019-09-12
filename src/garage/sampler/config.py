"""Class for configuring a Sampler."""

from collections import defaultdict

import numpy as np

from garage.experiment import deterministic


def rollout(worker_number, agent, env, max_path_length):
    """Sample a single rollout of the agent in the environment.

    Returns a tuple containing the following 2D arrays,
    with the first dimension corresponding to the time dimension:
        - observations
        - actions
        - rewards
        - next_observations
        - terminals
    The last two elements of the tuple will be dictionaries mapping strings to
    2D arrays:
        - agent_infos
        - env_infos
    """
    observations = []
    actions = []
    rewards = []
    agent_infos = defaultdict(list)
    env_infos = defaultdict(list)
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        actions.append(a)
        for k, v in agent_info.items():
            agent_infos[k].append(v)
        for k, v in env_info.items():
            env_infos[k].append(v)
        path_length += 1
        if d:
            break
        o = next_o
    for k, v in agent_infos.items():
        agent_infos[k] = np.asarray(v)
    for k, v in env_infos.items():
        env_infos[k] = np.asarray(v)
    return (worker_number, np.array(observations), np.array(actions),
            np.array(rewards), dict(agent_infos), dict(env_infos))


def default_env_update_fn(worker_number, env, env_update):
    """Use any non-None env_update as a new environment."""
    if env_update is not None:
        return env_update
    else:
        return env


def default_agent_update_fn(worker_number, agent, agent_update):
    """Update an agent, assuming it implements garage.tf.policies.Policy2."""
    agent.set_param_values(agent_update)
    return agent


class SamplerConfig:
    """An object for configuring the sampler.

    The intent is that this object should be sufficient to avoid subclassing
    the sampler. Instead of subclassing the sampler for e.g. a specific
    backend, implement a specialized SamplerConfig (or specify appropriate
    functions to this one).
    """

    def __init__(self,
                 *,
                 seed,
                 max_path_length,
                 worker_init_fn=None,
                 env_update_fn=default_env_update_fn,
                 agent_update_fn=default_agent_update_fn,
                 rollout_fn=rollout):
        self.seed = seed
        self.max_path_length = max_path_length
        if worker_init_fn is None:

            def default_worker_init_fn(worker_number):
                deterministic.set_seed(seed + worker_number)

            self.worker_init_fn = default_worker_init_fn
        else:
            self.worker_init_fn = worker_init_fn
        self.env_update_fn = env_update_fn
        self.agent_update_fn = agent_update_fn
        self.rollout_fn = rollout_fn
