"""A regression test for automatic benchmarking garage-PyTorch-A2C."""
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.torch.algos.a2c_2 import A2C
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from garage.torch.value_functions import ContinuousMLPValueFunction


@wrap_experiment(snapshot_mode='none')
def a2c_pendulum(ctxt, seed):
    """Create garage PyTorch A2C model and training.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the
            snapshotter.
        seed (int): Random positive integer for the trial.

    """
    deterministic.set_seed(seed)

    runner = LocalRunner(ctxt)

    env = GymEnv('InvertedPendulum-v2')

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=(32, 32),
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    value_function = ContinuousMLPValueFunction(env_spec=env.spec,
                                                hidden_sizes=(32, 32),
                                                hidden_nonlinearity=torch.tanh,
                                                output_nonlinearity=None)

    policy_optimizer = OptimizerWrapper((torch.optim.Adam, dict(lr=1e-3)),
                                        policy)

    vf_optimizer = OptimizerWrapper((torch.optim.Adam, dict(lr=1e-3)),
                                    value_function)

    algo = A2C(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        policy_optimizer=policy_optimizer,
        vf_optimizer=vf_optimizer,
        stop_entropy_gradient=True,
        center_adv=False,
        max_episode_length=env.spec.max_episode_length,
        discount=0.99,
    )

    runner.setup(algo, env)
    runner.train(n_epochs=2000, batch_size=2000)


a2c_pendulum(seed=0)
