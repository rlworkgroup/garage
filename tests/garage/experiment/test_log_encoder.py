import json

import numpy as np
import torch

from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.experiment.experiment import LogEncoder
from garage.plotter import Plotter
from garage.sampler import LocalSampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from tests.fixtures import snapshot_config


def test_encode_none_timedelta64():
    value = {'test': np.timedelta64(None)}
    encoded = json.dumps(value,
                         indent=2,
                         sort_keys=False,
                         cls=LogEncoder,
                         check_circular=False)
    assert 'test' in encoded


def test_encode_trainer():
    env = normalize(GymEnv('InvertedDoublePendulum-v2'))
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )
    value_function = GaussianMLPValueFunction(env_spec=env.spec)
    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length,
                           is_tf_worker=False)

    trainer = Trainer(snapshot_config)
    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               gae_lambda=0.97,
               lr_clip_range=2e-1)

    trainer.setup(algo, env)
    encoded = json.dumps(trainer,
                         indent=2,
                         sort_keys=False,
                         cls=LogEncoder,
                         check_circular=False)
    print(encoded)
    assert 'value_function' in encoded
