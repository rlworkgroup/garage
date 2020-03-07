"""Example script to run RL2 in HalfCheetah."""
from garage import wrap_experiment
from garage.envs.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.experiment import task_sampler
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.sampler.rl2_worker import RL2Worker
from garage.tf.algos import RL2
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianGRUPolicy


@wrap_experiment
def rl2_ppo_halfcheetah(ctxt=None, seed=1):
    """Train PPO with HalfCheetah environment.

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

        # ---- For ML1-push
        # env = RL2Env(ML1.get_train_tasks('push-v1'))
        # tasks = task_sampler.EnvPoolSampler([env])
        # tasks.grow_pool(meta_batch_size)

        # ---- For HalfCheetahVel
        tasks = task_sampler.SetTaskSampler(lambda: RL2Env(
            env=HalfCheetahVelEnv()))

        env_spec = tasks.sample(1)[0]().spec
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=64,
                                   env_spec=env_spec,
                                   state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=env_spec)

        inner_algo = RL2PPO(
            env_spec=env_spec,
            policy=policy,
            baseline=baseline,
            max_path_length=max_path_length * episode_per_task,
            discount=0.99,
            gae_lambda=0.95,
            lr_clip_range=0.2,
            optimizer_args=dict(
                batch_size=32,
                max_epochs=10,
            ),
            stop_entropy_gradient=True,
            entropy_method='max',
            policy_ent_coeff=0.02,
            center_adv=False,
        )

        algo = RL2(inner_algo=inner_algo,
                   max_path_length=max_path_length,
                   meta_batch_size=meta_batch_size,
                   task_sampler=tasks)

        runner.setup(algo,
                     tasks.sample(meta_batch_size),
                     sampler_cls=LocalSampler,
                     n_workers=meta_batch_size,
                     worker_class=RL2Worker)

        runner.train(n_epochs=n_epochs,
                     batch_size=episode_per_task * max_path_length *
                     meta_batch_size)


rl2_ppo_halfcheetah(seed=1)
