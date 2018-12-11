from garage.experiment import run_experiment
import garage.torch.algos.pytorch_util as ptu
from garage.torch.envs.wrappers import NormalizedBoxEnv
from garage.torch.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from garage.torch.exploration_strategies.ou_strategy import OUStrategy
from garage.torch.algos.modules import HuberLoss
from garage.torch.envs.tdm.half_cheetah_env import GoalXVelHalfCheetah
from garage.torch.algos.tdm.her_replay_buffer import HerReplayBuffer
from garage.torch.algos.tdm.networks import TdmNormalizer, TdmQf, TdmPolicy
from garage.torch.algos.tdm.tdm import TemporalDifferenceModel


def experiment(variant):
    env = NormalizedBoxEnv(GoalXVelHalfCheetah())
    max_tau = variant['tdm_kwargs']['max_tau']
    tdm_normalizer = TdmNormalizer(
        env,
        vectorized=True,
        max_tau=max_tau,
    )
    qf = TdmQf(
        env=env,
        vectorized=True,
        norm_order=1,
        tdm_normalizer=tdm_normalizer,
        hidden_sizes=[300, 300],
    )
    policy = TdmPolicy(
        env=env,
        tdm_normalizer=tdm_normalizer,
        hidden_sizes=[300, 300],
    )
    es = OUStrategy(
        action_space=env.action_space,
        theta=0.1,
        max_sigma=0.1,
        min_sigma=0.1,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        max_size=int(1E6),
    )
    algorithm = TemporalDifferenceModel(
        env,
        qf=qf,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        tdm_normalizer=tdm_normalizer,
        qf_criterion=HuberLoss(),
        **variant['tdm_kwargs'])
    algorithm.to(ptu.device)
    algorithm.train()


def run_task(*_):
    variant = dict(
        tdm_kwargs=dict(
            # TDM parameters
            max_tau=10,
            num_pretrain_paths=20,  # Set to zero to not normalize

            # General parameters
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=10000,
            max_path_length=99,
            num_updates_per_env_step=25,
            batch_size=128,
            discount=1,  # TDMs does no discounting
            reward_scale=10,

            # DDPG soft-target tau (not TDM tau)
            tau=0.001,
        ),
        algorithm="TDM",
    )
    experiment(variant)


run_experiment(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random
    # seed will be used
    seed=1,
    # plot=True,
)
