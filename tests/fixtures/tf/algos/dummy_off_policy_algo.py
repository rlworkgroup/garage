from garage.tf.algos import OffPolicyRLAlgorithm


class DummyOffPolicyAlgo(OffPolicyRLAlgorithm):
    def init_opt(self):
        pass
