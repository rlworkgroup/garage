from garage.envs import GymEnv
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler, ReplayBufferSampler

from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.policies import DummyPolicy


def test_doesnt_crash():
    env = GymEnv(DummyBoxEnv(obs_dim=(4, 4), action_dim=(2, 2)))
    policy = DummyPolicy(env.spec)

    local_sampler = LocalSampler(policy, env)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sampler = ReplayBufferSampler(env.spec,
                                  replay_buffer,
                                  local_sampler,
                                  outer_samples_per_inner_sample=10)
    samples = sampler.obtain_samples(0, 100, policy)
    assert len(samples.rewards) == 100
    samples = sampler.obtain_samples(0, 100, policy)
    assert len(samples.rewards) == 100
    # There's already some samples due to the shared replay buffer from above
    sampler2 = ReplayBufferSampler(env.spec, replay_buffer)
    samples2 = sampler2.obtain_samples(0, 100, policy)
    assert len(samples2.rewards) == 100
