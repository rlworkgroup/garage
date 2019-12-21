''' Generic sampler can be used with all RL algos #TODO: avnishna update description'''
# from garage.sampler.base import BaseSampler

class SimpleSampler(BaseSampler):
    """Generic sampler used for off policy sampling

    Args:
        agent(Policy): Agent used to select actions.
        env(gym.Env): Environment to perform actions in.
    Notes:
        - unlike the off policy vectorized sampler, simple sampler does not add
            transitions to the agent's replay buffer. That is the responsibility
            of the agent (potentially inside of its (the agent's) process samples
            function.
    """
    def __init__(self, env, agent):
        self._env = env
        self._agent = agent

    def obtain_samples(self, batch_size):
        samples_collected = 0
        rg = rollout_generator(self._env, self._agent)
        returned_samples = [None] * batch_size
        while samples_collected < batch_size:
            try:
                sample = rg.next()
                returned_samples[samples_collected] = sample
                samples_collected += 1
            except:
                rg = rollout_generator(self._env, self._agent)
        return returned_samples





def rollout_generator(env,
                      agent,
                      *,
                      max_path_length=np.inf,
                      animated=False,
                      speedup=1,
                      deterministic=False):
    """Sample a single rollout of the agent in the environment.

    Args:
        agent(Policy): Agent used to select actions.
        env(gym.Env): Environment to perform actions in.
        max_path_length(int): If the rollout reaches this many timesteps, it is
            terminated.
        animated(bool): If true, render the environment after each step.
        speedup(float): Factor by which to decrease the wait time between
            rendered steps. Only relevant, if animated == true.
        deterministic (bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.

    Yields:
        dict[str, np.ndarray or dict]: Dictionary, with keys:
            * observations(np.array): Non-flattened array of observations.
                There should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape (T + 1, S^*) (the
                unflattened state space of the current environment).
            * actions(np.array): Non-flattened array of actions. Should have
                shape (T, S^*) (the unflattened action space of the current
                environment).
            * rewards(np.array): Array of rewards of shape (T,) (1D array of
                length timesteps).
            * agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            * env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.

    """
    agent.reset()
    path_length = 0
    last_observation = env.reset()
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(last_observation)
        if deterministic:
            a = agent_info['mean']
        next_o, r, d, env_info = env.step(a)
        yield dict(
            observation=last_observation,
            next_observation=next_o,
            reward=r,
            done=d,
            agent_info=agent_info,
            env_info=env_info,
        )
        last_observation = next_o
        path_length += 1
        if d:
            break
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)

