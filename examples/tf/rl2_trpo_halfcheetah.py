"""Example script to run RL2 in HalfCheetah."""
from garage import wrap_experiment
from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.experiment import task_sampler
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2TRPO
from garage.tf.algos.rl2 import RL2Env
from garage.tf.algos.rl2 import RL2Worker
from garage.tf.experiment import LocalTFRunner
from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import FiniteDifferenceHvp
from garage.tf.policies import GaussianGRUPolicy


@wrap_experiment
def rl2_trpo_halfcheetah(ctxt=None, seed=1):
    """Train TRPO with HalfCheetah environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        max_path_length = 100
        meta_batch_size = 10
        n_epochs = 50
        episode_per_task = 4

        tasks = task_sampler.SetTaskSampler(lambda: RL2Env(
            env=HalfCheetahVelEnv()))

        env_spec = RL2Env(env=HalfCheetahVelEnv()).spec
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=64,
                                   env_spec=env_spec,
                                   state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=env_spec)

        algo = RL2TRPO(rl2_max_path_length=max_path_length,
                       meta_batch_size=meta_batch_size,
                       task_sampler=tasks,
                       env_spec=env_spec,
                       policy=policy,
                       baseline=baseline,
                       max_path_length=max_path_length * episode_per_task,
                       discount=0.99,
                       max_kl_step=0.01,
                       optimizer=ConjugateGradientOptimizer,
                       optimizer_args=dict(hvp_approach=FiniteDifferenceHvp(
                           base_eps=1e-5)))

        runner.setup(algo,
                     tasks.sample(meta_batch_size),
                     sampler_cls=LocalSampler,
                     n_workers=meta_batch_size,
                     worker_class=RL2Worker)

        runner.train(n_epochs=n_epochs,
                     batch_size=episode_per_task * max_path_length *
                     meta_batch_size)


rl2_trpo_halfcheetah(seed=1)
