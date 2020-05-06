"""A dummy algorithm fixture."""
from garage.np.algos import BatchPolopt


class DummyAlgo(BatchPolopt):
    """Dummy algo for test.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.np.policies.Policy): Policy.
        baseline (garage.np.baselines.Baseline): The baseline.

    """

    def __init__(self, env_spec, policy, baseline):
        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         discount=0.1,
                         max_path_length=1,
                         n_samples=10)

    def train_once(self, itr, paths):
        pass

    def train(self, runner):
        pass
