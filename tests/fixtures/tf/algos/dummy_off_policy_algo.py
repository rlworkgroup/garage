from garage.np.algos import OffPolicyRLAlgorithm


class DummyOffPolicyAlgo(OffPolicyRLAlgorithm):

    def init_opt(self):
        pass

    def train(self, runner):
        pass
