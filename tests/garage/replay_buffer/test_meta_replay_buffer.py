"""This is a script to test MetaReplayBuffer."""
import numpy as np

from garage.replay_buffer.meta_replay_buffer import MetaReplayBuffer
from garage.sampler import InPlaceSampler
from garage.tf.envs import TfEnv
from garage.torch.policies import DeterministicMLPPolicy
from tests.fixtures.envs.dummy import DummyBoxEnv


def test_all():
    """Test all methods."""
    env = TfEnv(DummyBoxEnv())
    policy = DeterministicMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))
    max_path_length = 10
    max_samples = 50
    max_trajs = 50

    sampler = InPlaceSampler(env, policy, max_path_length)

    paths, _ = sampler.obtain_samples(max_samples=max_samples,
                                      max_trajs=max_trajs,
                                      accum_context=False)

    replay_buffer = MetaReplayBuffer(100, env.observation_space.low.size,
                                     env.action_space.low.size)

    i = 0
    for path in paths:
        replay_buffer.add_path(path)
        i += max_path_length
        assert replay_buffer.size() == i

    replay_buffer.clear()
    assert replay_buffer.size() == 0

    for path in paths:
        replay_buffer.add_path(path)

    batch_size = 3
    indices = np.random.randint(0, replay_buffer.size(), batch_size)
    out = replay_buffer.sample_data(indices)
    assert len(out['observations']) == batch_size
    assert len(out['actions']) == batch_size
    assert len(out['rewards']) == batch_size
    assert len(out['terminals']) == batch_size
    assert len(out['next_observations']) == batch_size

    batch_size = 10
    out = replay_buffer.random_batch(batch_size)
    assert len(out['observations']) == batch_size
    assert len(out['actions']) == batch_size
    assert len(out['rewards']) == batch_size
    assert len(out['terminals']) == batch_size
    assert len(out['next_observations']) == batch_size

    out = replay_buffer.random_sequence(batch_size)
    assert len(out['observations']) == batch_size
    assert len(out['actions']) == batch_size
    assert len(out['rewards']) == batch_size
    assert len(out['terminals']) == batch_size
    assert len(out['next_observations']) == batch_size
