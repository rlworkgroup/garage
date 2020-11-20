# yapf: disable
import warnings

import numpy as np

from garage.np import (concat_tensor_dict_list, explained_variance_1d,
                       pad_batch_array, pad_tensor,
                       stack_and_pad_tensor_dict_list, stack_tensor_dict_list)

# yapf: enable

data = [
    dict(obs=[1, 1, 1], act=[2, 2, 2], info=dict(lala=[1, 1], baba=[2, 2])),
    dict(obs=[1, 1, 1], act=[2, 2, 2], info=dict(lala=[1, 1], baba=[2, 2]))
]
data2 = [
    dict(obs=[1, 1, 1], act=[2, 2, 2], info=dict(lala=[1, 1], baba=[2, 2])),
    dict(obs=[1, 1, 1], act=[2, 2, 2], info=dict(lala=[1, 1]))
]
max_len = 10
tensor = [1, 1, 1]


def test_concat_tensor_dict_list():
    results = concat_tensor_dict_list(data)
    assert results['obs'].shape == (6, )
    assert results['act'].shape == (6, )
    assert results['info']['lala'].shape == (4, )
    assert results['info']['baba'].shape == (4, )

    results = concat_tensor_dict_list(data2)
    assert results['obs'].shape == (6, )
    assert results['act'].shape == (6, )
    assert results['info']['lala'].shape == (4, )
    assert results['info']['baba'].shape == (2, )


def test_stack_tensor_dict_list():
    results = stack_tensor_dict_list(data)
    assert results['obs'].shape == (2, 3)
    assert results['act'].shape == (2, 3)
    assert results['info']['lala'].shape == (2, 2)
    assert results['info']['baba'].shape == (2, 2)

    results = stack_tensor_dict_list(data2)
    assert results['obs'].shape == (2, 3)
    assert results['act'].shape == (2, 3)
    assert results['info']['lala'].shape == (2, 2)
    assert results['info']['baba'].shape == (2, )


def test_pad_tensor():
    results = pad_tensor(tensor, max_len)
    assert len(tensor) == 3
    assert np.array_equal(results, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0])

    results = pad_tensor(tensor, max_len, mode='last')
    assert np.array_equal(results, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def test_explained_variance_1d():
    y = np.array([1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
    y_hat = np.array([2, 3, 4, 5, 6, 0, 0, 0, 0, 0])
    valids = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    result = explained_variance_1d(y, y_hat, valids)
    assert result == 1.0
    result = explained_variance_1d(y, y_hat)
    np.testing.assert_almost_equal(result, 0.95)


def test_stack_and_pad_tensor_dict_list():
    result = stack_and_pad_tensor_dict_list(data, max_len=5)
    assert np.array_equal(result['obs'],
                          np.array([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]]))
    assert np.array_equal(result['info']['lala'],
                          np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0]]))
    assert np.array_equal(result['info']['baba'],
                          np.array([[2, 2, 0, 0, 0], [2, 2, 0, 0, 0]]))


def test_pad_batch_array_warns_on_too_long():
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        result = pad_batch_array(np.ones(9), [5, 2, 2], 2)
        assert len(warns) == 1
        assert 'longer length than requested' in str(warns[0].message)
    assert (result == np.asarray([[1., 1., 1., 1., 1.], [1., 1., 0., 0., 0.],
                                  [1., 1., 0., 0., 0.]])).all()
