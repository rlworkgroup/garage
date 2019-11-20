import torch

from garage.envs import normalize
from garage.experiment import run_experiment
from garage.envs.base import GarageEnv
from garage.torch.envs import HalfCheetahVelEnv
from garage.experiment import LocalRunner
from garage.np.baselines import MultiTaskBaseline, LinearFeatureBaseline
from garage.torch.optimizers import ConjugateGradientOptimizer
from garage.torch.algos import VPG, MAML
from garage.torch.policies import GaussianMLPPolicy
from garage.sampler import BatchSampler
from garage.sampler import RaySampler

meta_batch_size = 40 # num_tasks
fast_batch_size = 20
learning_rate = 0.1
max_kl_step = 0.01
max_path_length = 200
n_parallel = 1

def run_task(snapshot_config, *_):
    env = GarageEnv(normalize(HalfCheetahVelEnv()))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 100),
        hidden_nonlinearity=torch.relu,
        output_nonlinearity=None,
    )
    '''
    baseline = MultiTaskBaseline(
        env_spec=env.spec,
        n_tasks=meta_batch_size,
        baseline_cls=LinearFeatureBaseline
    )
    '''

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    runner = LocalRunner(
        snapshot_config=snapshot_config,
        max_cpus=n_parallel)

    meta_optimizer = ConjugateGradientOptimizer(
                        policy.parameters(),
                        max_constraint_value=max_kl_step)

    inner_algo = VPG(env_spec=env.spec,
                policy=policy,
                baseline=baseline,
                max_path_length=max_path_length,
                discount=0.99,
                gae_lambda=1.0)

    algo = MAML(env=env,
                policy=policy,
                baseline=baseline,
                meta_batch_size=meta_batch_size,
                lr=learning_rate,
                inner_algo=inner_algo,
                num_grad_updates=1,
                meta_optimizer=meta_optimizer)

    runner.setup(algo=algo,
                 env=env)

    runner.train(n_epochs=500, batch_size=fast_batch_size * max_path_length)


run_experiment(
    run_task,
    snapshot_mode='all',
    seed=7,
    n_parallel=n_parallel
)