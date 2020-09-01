import numpy as np
import pytest

from garage.sampler import utils


class TestTruncatePaths:

    def setup_method(self):
        self.paths = [
            dict(
                observations=np.zeros((100, 1)),
                actions=np.zeros((100, 1)),
                rewards=np.zeros(100),
                env_infos=dict(),
                agent_infos=dict(lala=np.zeros(100)),
            ),
            dict(
                observations=np.zeros((50, 1)),
                actions=np.zeros((50, 1)),
                rewards=np.zeros(50),
                env_infos=dict(),
                agent_infos=dict(lala=np.zeros(50)),
            ),
        ]
        self.paths_dict = [
            dict(
                observations=np.zeros((100, 1)),
                actions=np.zeros((100, 1)),
                rewards=np.zeros(100),
                env_infos=dict(),
                agent_infos=dict(lala=dict(baba=np.zeros(100))),
            ),
            dict(
                observations=np.zeros((50, 1)),
                actions=np.zeros((50, 1)),
                rewards=np.zeros(50),
                env_infos=dict(),
                agent_infos=dict(lala=dict(baba=np.zeros(50))),
            ),
        ]

    def test_truncates(self):
        truncated = utils.truncate_paths(self.paths, 130)
        assert len(truncated) == 2
        assert len(truncated[-1]['observations']) == 30
        assert len(truncated[0]['observations']) == 100
        # make sure not to change the original one
        assert len(self.paths) == 2
        assert len(self.paths[-1]['observations']) == 50

    def test_truncates_dict(self):
        truncated = utils.truncate_paths(self.paths_dict, 130)
        assert len(truncated) == 2
        assert len(truncated[-1]['agent_infos']['lala']['baba']) == 30
        assert len(truncated[0]['agent_infos']['lala']['baba']) == 100

    def test_invalid_path(self):
        self.paths[0]['invalid'] = None
        with pytest.raises(ValueError):
            utils.truncate_paths(self.paths, 3)
