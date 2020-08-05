"""
This script creates a test that tests functions in garage.misc.tensor_utils.
"""
# yapf: disable
import numpy as np

from garage.envs import GymEnv
from garage.misc.tensor_utils import (concat_tensor_dict_list,
                                      explained_variance_1d,
                                      normalize_pixel_batch,
                                      pad_tensor,
                                      stack_and_pad_tensor_dict_list,
                                      stack_tensor_dict_list)

from tests.fixtures.envs.dummy import DummyDiscretePixelEnv

# yapf: enable


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

    def test_normalize_pixel_batch(self):
        env = GymEnv(DummyDiscretePixelEnv(), is_image=True)
        obs, _ = env.reset()
        obs_normalized = normalize_pixel_batch(obs)
        expected = [ob / 255.0 for ob in obs]
        assert np.allclose(obs_normalized, expected)

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

    def test_explained_variance_1d(self):
        y = np.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
        y_hat = np.array([2, 3, 4, 5, 6, 0, 0, 0, 0, 0])
        valids = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        result = explained_variance_1d(y, y_hat, valids)
        assert result == 1.0
        result = explained_variance_1d(y, y_hat)
        np.testing.assert_almost_equal(result, 0.95)

    def test_stack_and_pad_tensor_dict_list(self):
        result = stack_and_pad_tensor_dict_list(self.data, max_len=5)
        assert np.array_equal(result['obs'],
                              np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]]))
        assert np.array_equal(result['info']['lala'],
                              np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]))
        assert np.array_equal(result['info']['baba'],
                              np.array([[2, 2, 0, 0, 0], [2, 2, 0, 0, 0]]))
