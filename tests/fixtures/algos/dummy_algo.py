"""A dummy algorithm fixture."""
from garage.np.algos import BatchPolopt


class DummyAlgo(BatchPolopt):
    """Dummy algo for test.

    Args:
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
    """

    def __init__(self, policy, baseline):
        super().__init__(policy=policy,
                         baseline=baseline,
                         discount=0.1,
                         max_path_length=1,
                         n_samples=10)

    def train_once(self, itr, paths):
        pass

    def train(self, runner):
        pass
