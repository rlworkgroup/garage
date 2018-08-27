from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from garage.baselines import LinearFeatureBaseline
from garage.envs.env_spec import EnvSpec
from garage.envs.mujoco.sawyer import SimplePushEnv
from garage.misc.instrument import run_experiment
from garage.tf.spaces import Box

from sandbox.embed2learn.algos import PPOTaskEmbedding
from sandbox.embed2learn.baselines import MultiTaskLinearFeatureBaseline
from sandbox.embed2learn.baselines import MultiTaskGaussianMLPBaseline
from sandbox.embed2learn.envs import MultiTaskEnv
from sandbox.embed2learn.envs.multi_task_env import TfEnv
from sandbox.embed2learn.embeddings import EmbeddingSpec
from sandbox.embed2learn.embeddings import GaussianMLPEmbedding
from sandbox.embed2learn.embeddings.utils import concat_spaces
from sandbox.embed2learn.policies import GaussianMLPMultitaskPolicy


DELTAS = [
    np.array([ 0.15,    0., 0.]),
    np.array([-0.15,    0., 0.]),
    np.array([   0.,  0.15, 0.]),
    np.array([   0., -0.15, 0.]),
]

TASKS = {
    str(i + 1): {
        "args": [],
        "kwargs": {
            "control_method": "position_control",
            "delta": d,
            "completion_bonus": 0.0,
            "action_scale": 0.04,
            "randomize_start_jpos": False,
            "collision_penalty": 2,
        }
    }
    for i, d in enumerate(DELTAS)
}


def run_task(v):
    v = SimpleNamespace(**v)

    v.tasks = TASKS
    task_names = sorted(v.tasks.keys())
    task_args = [v.tasks[t]['args'] for t in task_names]
    task_kwargs = [v.tasks[t]['kwargs'] for t in task_names]

    # Environment
    env = TfEnv(
            MultiTaskEnv(
                task_env_cls=SimplePushEnv,
                task_args=task_args,
                task_kwargs=task_kwargs))

    # Latent space and embedding specs
    # TODO(gh/10): this should probably be done in Embedding or Algo
    latent_lb = np.zeros(v.latent_length, )
    latent_ub = np.ones(v.latent_length, )
    latent_space = Box(latent_lb, latent_ub)

    # trajectory space is (TRAJ_ENC_WINDOW, act_obs) where act_obs is a stacked
    # vector of flattened actions and observations
    act_lb, act_ub = env.action_space.bounds
    act_lb_flat = env.action_space.flatten(act_lb)
    act_ub_flat = env.action_space.flatten(act_ub)
    obs_lb, obs_ub = env.observation_space.bounds
    obs_lb_flat = env.observation_space.flatten(obs_lb)
    obs_ub_flat = env.observation_space.flatten(obs_ub)
    # act_obs_lb = np.concatenate([act_lb_flat, obs_lb_flat])
    # act_obs_ub = np.concatenate([act_ub_flat, obs_ub_flat])
    act_obs_lb = obs_lb_flat
    act_obs_ub = obs_ub_flat
    # act_obs_lb = act_lb_flat
    # act_obs_ub = act_ub_flat
    traj_lb = np.stack([act_obs_lb] * v.inference_window)
    traj_ub = np.stack([act_obs_ub] * v.inference_window)
    traj_space = Box(traj_lb, traj_ub)

    task_embed_spec = EmbeddingSpec(env.task_space, latent_space)
    traj_embed_spec = EmbeddingSpec(traj_space, latent_space)
    task_obs_space = concat_spaces(env.task_space, env.observation_space)
    env_spec_embed = EnvSpec(task_obs_space, env.action_space)

    # TODO(): rename to inference_network
    traj_embedding = GaussianMLPEmbedding(
        name="inference",
        embedding_spec=traj_embed_spec,
        hidden_sizes=(256, 128),  # was the same size as policy in Karol's paper
        std_share_network=True,
        init_std=v.embedding_init_std,
        # output_nonlinearity=tf.tanh,
        max_std=v.embedding_max_std,
    )

    # Embeddings
    task_embedding = GaussianMLPEmbedding(
        name="embedding",
        embedding_spec=task_embed_spec,
        hidden_sizes=(256, 256),
        std_share_network=True,
        init_std=v.embedding_init_std,
        max_std=v.embedding_max_std,
        # output_nonlinearity=tf.tanh,
    )

    # Multitask policy
    policy = GaussianMLPMultitaskPolicy(
        name="policy",
        env_spec=env.spec,
        task_space=env.task_space,
        embedding=task_embedding,
        hidden_sizes=(256, 128),
        std_share_network=True,
        init_std=v.policy_init_std,
        max_std=v.policy_max_std,
    )

    extra = v.latent_length + len(v.tasks)
    baseline = MultiTaskGaussianMLPBaseline(env_spec=env.spec,
                                            extra_dims=extra,
                                            regressor_args=dict(hidden_sizes=(256, 128)))

    algo = PPOTaskEmbedding(
        env=env,
        policy=policy,
        baseline=baseline,
        inference=traj_embedding,
        batch_size=v.batch_size,
        max_path_length=v.max_path_length,
        n_itr=6000,
        discount=0.99,
        step_size=0.2,
        plot=True,
        policy_ent_coeff=v.policy_ent_coeff,
        embedding_ent_coeff=v.embedding_ent_coeff,
        inference_ce_coeff=v.inference_ce_coeff,
        use_softplus_entropy=True,
	save_sample_frequency=50
    )
    algo.train()


config = dict(
    latent_length=3,
    inference_window=15,
    batch_size=4096 * len(TASKS),
    policy_ent_coeff=5e-3,  # 1e-2
    embedding_ent_coeff=2e-3,  # 1e-3
    inference_ce_coeff=1.25e-4,  # 1e-4
    max_path_length=400,
    embedding_init_std=0.4,
    embedding_max_std=1,
    policy_init_std=1,
    policy_max_std=None,
)

run_experiment(
    run_task,
    exp_prefix='sawyer_pusher_rel_obs_embed_udlr',
    n_parallel=8,
    seed=1,
    variant=config,
    plot=False,
)
