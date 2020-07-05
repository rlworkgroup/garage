from dowel import logger
import numpy as np

from garage.sampler.utils import truncate_paths

from tests.fixtures.logger import NullOutput


class TestSampler:

    def setup_method(self):
        logger.add_output(NullOutput())

    def teardown_method(self):
        logger.remove_all()

    def test_truncate_paths(self):
        paths = [
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

        truncated = truncate_paths(paths, 130)
        assert len(truncated) == 2
        assert len(truncated[-1]['observations']) == 30
        assert len(truncated[0]['observations']) == 100
        # make sure not to change the original one
        assert len(paths) == 2
        assert len(paths[-1]['observations']) == 50
