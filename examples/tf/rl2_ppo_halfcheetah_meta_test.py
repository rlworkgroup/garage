"""Example script to run RL2PPO meta test in HalfCheetah."""
from garage import wrap_experiment
from garage.envs import HalfCheetahVelEnv
from garage.experiment import task_sampler
from garage.experiment.deterministic import set_seed
from garage.experiment.meta_evaluator import MetaEvaluator
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env
from garage.tf.algos.rl2 import RL2Worker
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import GaussianGRUPolicy


@wrap_experiment
def rl2_ppo_halfcheetah_meta_test(ctxt=None, seed=1):
    """Perform meta-testing on RL2PPO with HalfCheetah environment.

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

        meta_evaluator = MetaEvaluator(test_task_sampler=tasks,
                                       n_exploration_traj=10,
                                       n_test_rollouts=10,
                                       max_path_length=max_path_length,
                                       n_test_tasks=5)

        algo = RL2PPO(rl2_max_path_length=max_path_length,
                      meta_batch_size=meta_batch_size,
                      task_sampler=tasks,
                      env_spec=env_spec,
                      policy=policy,
                      baseline=baseline,
                      discount=0.99,
                      gae_lambda=0.95,
                      lr_clip_range=0.2,
                      pg_loss='surrogate_clip',
                      optimizer_args=dict(
                          batch_size=32,
                          max_epochs=10,
                      ),
                      stop_entropy_gradient=True,
                      entropy_method='max',
                      policy_ent_coeff=0.02,
                      center_adv=False,
                      max_path_length=max_path_length * episode_per_task,
                      meta_evaluator=meta_evaluator,
                      n_epochs_per_eval=10)

        runner.setup(algo,
                     tasks.sample(meta_batch_size),
                     sampler_cls=LocalSampler,
                     n_workers=meta_batch_size,
                     worker_class=RL2Worker)

        runner.train(n_epochs=n_epochs,
                     batch_size=episode_per_task * max_path_length *
                     meta_batch_size)


rl2_ppo_halfcheetah_meta_test(seed=1)
