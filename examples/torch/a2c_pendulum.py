"""A regression test for automatic benchmarking garage-PyTorch-A2C."""
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.torch.algos import A2C
from garage.torch.optimizers import OptimizerWrapper
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from garage.torch.value_functions import ContinuousMLPValueFunction


@wrap_experiment
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

    env = normalize(GymEnv('InvertedPendulum-v2'))

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes=(32, 32),
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    value_function = ContinuousMLPValueFunction(env_spec=env.spec,
                                                hidden_sizes=(32, 32),
                                                hidden_nonlinearity=torch.tanh,
                                                output_nonlinearity=None)

    policy_optimizer = OptimizerWrapper(
        (torch.optim.Adam, dict(lr=1e-3)), policy)

    vf_optimizer = OptimizerWrapper(
        (torch.optim.Adam, dict(lr=1e-3)),
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
    runner.train(n_epochs=1000,
                 batch_size=500)


a2c_pendulum(seed=1)
