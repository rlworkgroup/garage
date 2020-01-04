"""This is a script to test InPlaceSampler."""

from garage.sampler import InPlaceSampler
from garage.tf.envs import TfEnv
from garage.torch.policies import DeterministicMLPPolicy
from tests.fixtures.envs.dummy import DummyBoxEnv


def test_obtain_samples():
    """Test obtain_samples method."""
    env = TfEnv(DummyBoxEnv())
    policy = DeterministicMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))
    max_path_length = 10
    max_samples = 20
    max_trajs = 20

    sampler = InPlaceSampler(env, policy, max_path_length)

    paths, steps = sampler.obtain_samples(max_samples=max_samples,
                                          max_trajs=max_trajs,
                                          accum_context=False)

    total_steps = 0
    obs_dim = len(paths[0]['observations'][0])
    act_dim = len(paths[0]['actions'][0])
    for path in paths:
        path_length = len(path['observations'])
        total_steps += path_length

    assert (obs_dim, ) == env.observation_space.shape
    assert (act_dim, ) == env.action_space.shape
    assert path_length == max_path_length
    assert total_steps == max_samples
    assert steps == max_samples
