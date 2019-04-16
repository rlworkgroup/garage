from garage.core import Serializable

class Algorithm:
    pass


class RLAlgorithm(Algorithm, Serializable):
    def train(self):
        raise NotImplementedError
