"""A dummy experiment fixture."""
from garage.envs import GymEnv
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import VPG
from garage.tf.policies import CategoricalMLPPolicy
from garage.trainer import TFTrainer


# pylint: disable=missing-return-type-doc
def fixture_exp(snapshot_config, sess):
    """Dummy fixture experiment function.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by Trainer to create the snapshotter.
            If None, it will create one with default settings.
        sess (tf.Session): An optional TensorFlow session.
              A new session will be created immediately if not provided.

    Returns:
        np.ndarray: Values of the parameters evaluated in
            the current session

    """
    with TFTrainer(snapshot_config=snapshot_config, sess=sess) as trainer:
        env = GymEnv('CartPole-v1', max_episode_length=100)

        policy = CategoricalMLPPolicy(name='policy',
                                      env_spec=env.spec,
                                      hidden_sizes=(8, 8))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length)

        algo = VPG(env_spec=env.spec,
                   policy=policy,
                   baseline=baseline,
                   sampler=sampler,
                   discount=0.99,
                   optimizer_args=dict(learning_rate=0.01, ))

        trainer.setup(algo, env)
        trainer.train(n_epochs=5, batch_size=100)

        return policy.get_param_values()
