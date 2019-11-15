"""
This script creates a test that tests functions in garage.misc.tensor_utils.
"""

import numpy as np

from garage.misc.tensor_utils import concat_tensor_dict_list
from garage.misc.tensor_utils import normalize_pixel_batch
from garage.misc.tensor_utils import pad_tensor
from garage.misc.tensor_utils import stack_tensor_dict_list
from garage.tf.envs import TfEnv
from tests.fixtures.envs.dummy import DummyBoxEnv
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestTensorUtil:

    def setup_method(self):
        self.data = [
            dict(obs=[1, 1, 1],
                 act=[2, 2, 2],
                 info=dict(lala=[1, 1], baba=[2, 2])),
            dict(obs=[1, 1, 1],
                 act=[2, 2, 2],
                 info=dict(lala=[1, 1], baba=[2, 2]))
        ]
        self.data2 = [
            dict(obs=[1, 1, 1],
                 act=[2, 2, 2],
                 info=dict(lala=[1, 1], baba=[2, 2])),
            dict(obs=[1, 1, 1], act=[2, 2, 2], info=dict(lala=[1, 1]))
        ]
        self.max_len = 10
        self.tensor = [1, 1, 1]

    def test_normalize_pixel_patch(self):
        env = TfEnv(DummyDiscretePixelEnv())
        obs = env.reset()
        obs_normalized = normalize_pixel_batch(env, obs)
        expected = [ob / 255.0 for ob in obs]
        assert np.allclose(obs_normalized, expected)

    def test_normalize_pixel_patch_not_trigger(self):
        env = TfEnv(DummyBoxEnv())
        obs = env.reset()
        obs_normalized = normalize_pixel_batch(env, obs)
        assert np.array_equal(obs, obs_normalized)

    def test_concat_tensor_dict_list(self):
        results = concat_tensor_dict_list(self.data)
        assert results['obs'].shape == (6, )
        assert results['act'].shape == (6, )
        assert results['info']['lala'].shape == (4, )
        assert results['info']['baba'].shape == (4, )

        results = concat_tensor_dict_list(self.data2)
        assert results['obs'].shape == (6, )
        assert results['act'].shape == (6, )
        assert results['info']['lala'].shape == (4, )
        assert results['info']['baba'].shape == (2, )

    def test_stack_tensor_dict_list(self):
        results = stack_tensor_dict_list(self.data)
        assert results['obs'].shape == (2, 3)
        assert results['act'].shape == (2, 3)
        assert results['info']['lala'].shape == (2, 2)
        assert results['info']['baba'].shape == (2, 2)

        results = stack_tensor_dict_list(self.data2)
        assert results['obs'].shape == (2, 3)
        assert results['act'].shape == (2, 3)
        assert results['info']['lala'].shape == (2, 2)
        assert results['info']['baba'].shape == (2, )

    def test_pad_tensor(self):
        results = pad_tensor(self.tensor, self.max_len)
        assert len(self.tensor) == 3
        assert np.array_equal(results, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

        results = pad_tensor(self.tensor, self.max_len, mode='last')
        assert np.array_equal(results, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
