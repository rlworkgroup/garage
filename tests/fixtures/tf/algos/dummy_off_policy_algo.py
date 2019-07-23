from garage.np.algos import OffPolicyRLAlgorithm


class DummyOffPolicyAlgo(OffPolicyRLAlgorithm):
    def init_opt(self):
        pass

    def train_once(self, itr, paths):
        pass

    def train(self, runner, batch_size):
        pass
