"""A regression test for automatic benchmarking garage-TensorFlow-VPG."""
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.algos import VPG as TF_VPG
from garage.tf.policies import GaussianMLPPolicy as TF_GMP
from garage.trainer import TFTrainer

hyper_parameters = {
    'hidden_sizes': [64, 64],
    'center_adv': True,
    'learning_rate': 1e-2,
    'discount': 0.99,
    'n_epochs': 250,
    'batch_size': 2048,
}


@wrap_experiment
def vpg_garage_tf(ctxt, env_id, seed):
    """Create garage TensorFlow VPG model and training.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the
            snapshotter.
        env_id (str): Environment id of the task.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    with TFTrainer(ctxt) as trainer:
        env = normalize(GymEnv(env_id))

        policy = TF_GMP(
            env_spec=env.spec,
            hidden_sizes=hyper_parameters['hidden_sizes'],
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=None,
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = TF_VPG(env_spec=env.spec,
                      policy=policy,
                      baseline=baseline,
                      sampler=sampler,
                      discount=hyper_parameters['discount'],
                      center_adv=hyper_parameters['center_adv'],
                      optimizer_args=dict(
                          learning_rate=hyper_parameters['learning_rate'], ))

        trainer.setup(algo, env)
        trainer.train(n_epochs=hyper_parameters['n_epochs'],
                      batch_size=hyper_parameters['batch_size'])
