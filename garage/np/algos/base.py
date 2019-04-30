from garage.core import Serializable


class Algorithm:
    pass


class RLAlgorithm(Algorithm):
    def train(self):
        raise NotImplementedError
