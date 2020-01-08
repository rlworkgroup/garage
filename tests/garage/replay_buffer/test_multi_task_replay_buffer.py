"""This is a script to test MultiTaskReplayBuffer."""
from garage.replay_buffer.multi_task_replay_buffer import MultiTaskReplayBuffer
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

    tasks = [0, 1, 2]
    replay_buffer = MultiTaskReplayBuffer(100, env, tasks)

    print(paths[0].keys())
    print(paths[0]['observations'])
    print(replay_buffer)

    i = 0
    for t in tasks:
        for path in paths:
            replay_buffer.add_path(t, path)
            replay_buffer.terminate_episode(t)
            i += max_path_length
            assert replay_buffer.num_steps_can_sample(t) == i
        i = 0

    for t in tasks:
        replay_buffer.clear_buffer(t)
        assert replay_buffer.num_steps_can_sample(t) == 0

    for t in tasks:
        replay_buffer.add_paths(t, paths)
        assert replay_buffer.num_steps_can_sample(t) == max_samples

    batch_size = 5
    for t in tasks:
        out = replay_buffer.random_batch(t, batch_size, sequence=False)
        assert len(out['observations']) == batch_size
        assert len(out['actions']) == batch_size
        assert len(out['rewards']) == batch_size
        assert len(out['terminals']) == batch_size
        assert len(out['next_observations']) == batch_size
        out = replay_buffer.random_batch(t, batch_size, sequence=True)
        assert len(out['observations']) == batch_size
        assert len(out['actions']) == batch_size
        assert len(out['rewards']) == batch_size
        assert len(out['terminals']) == batch_size
        assert len(out['next_observations']) == batch_size
