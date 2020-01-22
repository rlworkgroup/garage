""" Generic sampler can be used with all RL algos #TODO: avnishna update description"""
import time

import numpy as np

from garage import TimeStep
from garage.sampler.base import BaseSampler

class SimpleSampler(BaseSampler):
    """Generic sampler used for off policy sampling

    Args:
        agent(Policy): Agent used to select actions.
        env(gym.Env): Environment to perform actions in.
        max_path_length(int): If the rollout reaches this many timesteps, it is
            terminated.
        deterministic (bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.
    Notes:
        - unlike the off policy vectorized sampler, simple sampler does not add
            transitions to the agent's replay buffer. That is the responsibility
            of the agent (potentially inside of its (the agent's) process samples
            function.
    """
    def __init__(self, algo, env, agent, *, max_path_length, deterministic=False):
        self._env = env
        self._agent = agent
        self._deterministic = deterministic
        self._max_path_length = max_path_length

    def obtain_samples(self, itr, batch_size):
        """ Obtain batch_size number of samples from the env.

        Args:
            batch_size(int): the number of samples to be collected.
        Returns:
            List of 
        """
        samples_collected = 0
        rg = rollout_generator(self._env,
                               self._agent,
                               deterministic=self._deterministic,
                               max_path_length=self._max_path_length)
        returned_samples = []
        while samples_collected < batch_size:
                rg = rollout_generator(self._env,
                               self._agent,
                               deterministic=self._deterministic,
                               max_path_length=self._max_path_length)
                for sample in rg:
                    if samples_collected >= batch_size:
                        break
                    returned_samples.append(sample)
                    samples_collected += 1
        assert len(returned_samples) == samples_collected == batch_size
        return returned_samples
    
    def shutdown_worker(self):
        pass

    def start_worker(self):
        pass



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
       Garage.TimeStep

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
        yield TimeStep(env_spec=env.spec,
                       observation=last_observation,
                       action=a,
                       reward=r,
                       next_observation=next_o,
                       terminal=d,
                       env_info=env_info,
                       agent_info=agent_info)
        last_observation = next_o
        path_length += 1
        if d:
            break
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)

