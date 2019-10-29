from garage.np.algos.base import RLAlgorithm


class NOP(RLAlgorithm):
    """NOP (no optimization performed) policy search algorithm"""

    def __init__(self, **kwargs):
        super(NOP, self).__init__(**kwargs)

    def init_opt(self):
        pass

    def optimize_policy(self, itr, samples_data):
        pass

    def get_itr_snapshot(self, itr, samples_data):
        return dict()

    def train(self):
        pass
