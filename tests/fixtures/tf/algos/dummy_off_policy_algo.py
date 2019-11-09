from garage.np.algos import OffPolicyRLAlgorithm


class DummyOffPolicyAlgo(OffPolicyRLAlgorithm):

    def init_opt(self):
        pass

    def train(self, runner):
        pass

    def train_once(self, itr, paths):
        pass

    def optimize_policy(self, itr, samples_data):
        pass
