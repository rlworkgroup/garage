from garage.tf.algos import BatchPolopt


class DummyTFAlgo(BatchPolopt):
    """Dummy algo for test."""

    def __init__(self, env_spec, policy, baseline):
        super().__init__(env_spec=env_spec, policy=policy, baseline=baseline)

    def init_opt(self):
        pass

    def optimize_policy(self, itr, samples_data):
        pass

    def get_itr_snapshot(self, itr):
        pass
