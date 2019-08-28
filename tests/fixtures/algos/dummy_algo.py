from garage.np.algos import BatchPolopt


class DummyAlgo(BatchPolopt):
    """Dummy algo for test."""

    def __init__(self, policy, baseline):
        super().__init__(policy=policy,
                         baseline=baseline,
                         discount=0.1,
                         max_path_length=1,
                         n_samples=10)

    def train(self, runner):
        pass
