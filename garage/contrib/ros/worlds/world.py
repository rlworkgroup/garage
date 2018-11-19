import os.path as osp


class World:
    MODEL_DIR = osp.join(osp.dirname(__file__), 'models')

    def initialize(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        raise NotImplementedError
