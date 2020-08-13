"""This is a script to test PEARLWorker."""

import akro
import numpy as np
import torch.nn as nn
from torch.nn import functional as F  # NOQA

from garage import EnvSpec
from garage.envs import GymEnv
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import (ContextConditionedPolicy,
                                   TanhGaussianMLPPolicy)

from tests.fixtures.envs.dummy import DummyBoxEnv


def test_methods():
    """Test PEARLWorker methods."""
    env_spec = GymEnv(DummyBoxEnv())
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

    max_episode_length = 20
    worker1 = PEARLWorker(seed=1,
                          max_episode_length=max_episode_length,
                          worker_number=1)
    worker1.update_agent(context_policy)
    worker1.update_env(env_spec)
    episodes = worker1.rollout()

    assert episodes.observations.shape == (max_episode_length, obs_dim)
    assert episodes.actions.shape == (max_episode_length, action_dim)
    assert episodes.rewards.shape == (max_episode_length, )

    worker2 = PEARLWorker(seed=1,
                          max_episode_length=max_episode_length,
                          worker_number=1,
                          deterministic=True,
                          accum_context=True)
    worker2.update_agent(context_policy)
    worker2.update_env(env_spec)
    episodes = worker2.rollout()

    assert context_policy.context.shape == (1, max_episode_length,
                                            encoder_input_dim)
    assert episodes.observations.shape == (max_episode_length, obs_dim)
    assert episodes.actions.shape == (max_episode_length, action_dim)
    assert episodes.rewards.shape == (max_episode_length, )
