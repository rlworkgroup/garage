"""This is a script to test PEARLSampler."""

import akro
import numpy as np
import torch.nn as nn
from torch.nn import functional as F  # NOQA

from garage.envs.env_spec import EnvSpec
from garage.sampler import PEARLSampler
from garage.tf.envs import TfEnv
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import ContextConditionedPolicy
from garage.torch.policies import TanhGaussianMLPPolicy
from tests.fixtures.envs.dummy import DummyBoxEnv


def test_obtain_samples():
    """Test obtain_samples method."""
    env_spec = TfEnv(DummyBoxEnv())
    latent_dim = 5
    latent_space = akro.Box(low=-1,
                            high=1,
                            shape=(latent_dim, ),
                            dtype=np.float32)

    # add latent space to observation space to create a new space
    augmented_obs_space = akro.Tuple(
        (env_spec.observation_space, latent_space))
    augmented_env_spec = EnvSpec(augmented_obs_space, env_spec.action_space)

    obs_dim = int(np.prod(env_spec.observation_space.shape))
    action_dim = int(np.prod(env_spec.action_space.shape))
    reward_dim = 1
    encoder_input_dim = obs_dim + action_dim + reward_dim
    encoder_output_dim = latent_dim * 2
    encoder_hidden_sizes = (3, 2, encoder_output_dim)

    context_encoder = MLPEncoder(input_dim=encoder_input_dim,
                                 output_dim=encoder_output_dim,
                                 hidden_nonlinearity=None,
                                 hidden_sizes=encoder_hidden_sizes,
                                 hidden_w_init=nn.init.ones_,
                                 output_w_init=nn.init.ones_)

    policy = TanhGaussianMLPPolicy(env_spec=augmented_env_spec,
                                   hidden_sizes=(3, 5, 7),
                                   hidden_nonlinearity=F.relu,
                                   output_nonlinearity=None)

    context_policy = ContextConditionedPolicy(latent_dim=latent_dim,
                                              context_encoder=context_encoder,
                                              policy=policy,
                                              use_information_bottleneck=True,
                                              use_next_obs=False)

    max_path_length = 10
    max_samples = 20
    max_trajs = 20

    sampler = PEARLSampler(env_spec, context_policy, max_path_length)

    sampler.start_worker()

    paths, steps = sampler.obtain_samples(max_samples=max_samples,
                                          max_trajs=max_trajs,
                                          accum_context=False)

    total_steps = 0
    obs_dim = len(paths[0]['observations'][0])
    act_dim = len(paths[0]['actions'][0])
    for path in paths:
        path_length = len(path['observations'])
        total_steps += path_length

    assert (obs_dim, ) == env_spec.observation_space.shape
    assert (act_dim, ) == env_spec.action_space.shape
    assert path_length == max_path_length
    assert total_steps == max_samples
    assert steps == max_samples

    paths, steps = sampler.obtain_samples(max_samples=max_samples,
                                          max_trajs=max_trajs,
                                          accum_context=True)

    assert (1, max_samples, encoder_input_dim) == context_policy.context.shape

    sampler.shutdown_worker()
