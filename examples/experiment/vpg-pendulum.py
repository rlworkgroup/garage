"""Basic example of Experiment."""
import gym

from garage.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.experiment import Experiment
from garage.experiment.checkpointers import DiskCheckpointer
from garage.experiment.core.misc import get_env_spec
from garage.experiment.loggers import BasicLogger
from garage.experiment.samplers import BatchSampler
from garage.experiment.torch.algos import VPG
from garage.experiment.torch.policies import GaussianMLPPolicy


env = gym.make('Pendulum-v0')
env_spec = get_env_spec(env)

sampler = BatchSampler(env=env, max_path_length=100)

policy = GaussianMLPPolicy(env_spec=env_spec, hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env_spec)

agent = VPG(env_spec=env_spec, policy=policy, baseline=baseline, discount=0.99)

# Alternatives: HDFS, S3, etc.
checkpointer = DiskCheckpointer(
    exp_dir='garage-experiment-pendulum', resume=True, prefix='experiment')

# Alternativs: Tensorboard, Plotter
logger = BasicLogger()

# Initialize or load checkpoint from exp_dir.
#
# If get interrupted and restarted,
# run_experiment will resume from last checkpoint.
#
# /exp_dir
#     prefix_timestamp_agent.pkl
#     prefix_timestamp_sampler.pkl

exp = Experiment(
    env=env,
    sampler=sampler,
    agent=agent,
    checkpointer=checkpointer,
    logger=logger,
    # experiment variant
    n_itr=40,
    batch_size=4000,
)

exp.train()
