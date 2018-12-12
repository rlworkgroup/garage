from garage.experiment import run_experiment
from garage.torch.algos.modules import HuberLoss
import garage.torch.algos.pytorch_util as ptu
from garage.torch.algos.tdm.her_replay_buffer import HerReplayBuffer
from garage.torch.algos.tdm.networks import TdmNormalizer, TdmPolicy, TdmQf
from garage.torch.algos.tdm.tdm import TemporalDifferenceModel
from garage.torch.envs.tdm.ant_env import GoalXYPosAnt
from garage.torch.envs.wrappers import NormalizedBoxEnv
from garage.torch.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from garage.torch.exploration_strategies.ou_strategy import OUStrategy


def experiment(variant):
    env = NormalizedBoxEnv(GoalXYPosAnt(max_distance=6))
    max_tau = variant['tdm_kwargs']['max_tau']
    # Normalizer isn't used unless you set num_pretrain_paths > 0
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
        qf_criterion=HuberLoss(),
        tdm_normalizer=tdm_normalizer,
        **variant['tdm_kwargs'])
    algorithm.to(ptu.device)
    algorithm.train()


def run_task(*_):
    variant = dict(
        tdm_kwargs=dict(
            # TDM parameters
            max_tau=49,
            num_pretrain_paths=0,

            # General parameters
            num_epochs=500,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=50,
            num_updates_per_env_step=5,
            batch_size=128,
            discount=1,
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
