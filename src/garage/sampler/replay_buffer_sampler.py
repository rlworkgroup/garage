from dataclasses import replace
import math

from garage import log_performance, TimeStepBatch


class ReplayBufferSampler:
    """Sampler that samples from a replay buffer.

    Optionally can contain an inner sampler that samples from a real
    environment.
    """

    def __init__(self,
                 env_spec,
                 replay_buffer,
                 inner_sampler=None,
                 outer_samples_per_inner_sample=None,
                 ignore_agent_updates=True):
        self._env_spec = env_spec
        self.replay_buffer = replay_buffer
        self.inner_sampler = inner_sampler
        self._outer_samples_per_inner_sample = outer_samples_per_inner_sample
        self._outer_samples = 0
        self._inner_samples = 0
        self.total_env_steps = 0
        self._ignore_agent_updates = ignore_agent_updates
        if inner_sampler is not None:
            assert self._outer_samples_per_inner_sample is not None
        if self._outer_samples_per_inner_sample is not None:
            assert inner_sampler is not None

    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        self._outer_samples += num_samples
        if self.inner_sampler is not None:
            expected_inner_samples = (self._outer_samples /
                                      self._outer_samples_per_inner_sample)
            needed_samples = int(
                math.ceil(expected_inner_samples - self._inner_samples))
            if needed_samples > 0:
                self.fill_buffer(itr, needed_samples, agent_update, env_update)
        else:
            self.total_env_steps += num_samples
        return self.replay_buffer.sample_timesteps(num_samples)

    def fill_buffer(self, itr, samples_to_add, agent_update, env_update=None):
        """Fill the buffer by sampling from the inner sampler."""
        assert self.inner_sampler is not None
        if self._ignore_agent_updates:
            agent_update = None
        traj = self.inner_sampler.obtain_samples(0,
                                                 samples_to_add,
                                                 agent_update=agent_update,
                                                 env_update=env_update)
        if 'masked_observations' in traj.env_infos:
            # This isn't really correct if masked_observations is actually a
            # different type of observation.
            # assert traj.env_infos['masked_observations'].shape == traj.observations.shape
            traj = replace(traj,
                           observations=traj.env_infos['masked_observations'])
        if 'action_targets' in traj.agent_infos:
            assert traj.agent_infos[
                'action_targets'].shape == traj.actions.shape
            traj = replace(traj, actions=traj.agent_infos['action_targets'])
        self._inner_samples += len(traj.rewards)
        self.total_env_steps = self.inner_sampler.total_env_steps
        self.replay_buffer.add_episode_batch(traj)
        log_performance(itr,
                        traj,
                        discount=1.0,
                        prefix='ReplayBufferSamplerExpert')

    def shutdown_worker(self):
        """Shutdown the inner sampler."""
        if self.inner_sampler is not None:
            self.inner_sampler.shutdown_worker()
