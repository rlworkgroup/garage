"""Test root level functions in garage."""
# yapf: disable
import csv
import math
import tempfile

import akro
import dowel
from dowel import logger, tabular
import numpy as np
import pytest
import tensorflow as tf
import torch

from garage import (_Default,
                    EnvSpec,
                    EpisodeBatch,
                    log_multitask_performance,
                    log_performance,
                    make_optimizer,
                    StepType)

from tests.fixtures import TfGraphTestCase

# yapf: enable


@pytest.mark.serial
def test_log_performance():
    lengths = np.array([10, 5, 1, 1])
    batch = EpisodeBatch(
        EnvSpec(akro.Box(np.array([0., 0., 0.]), np.array([1., 1., 1.])),
                akro.Box(np.array([-1., -1.]), np.array([0., 0.]))),
        observations=np.ones((sum(lengths), 3), dtype=np.float32),
        last_observations=np.ones((len(lengths), 3), dtype=np.float32),
        actions=np.zeros((sum(lengths), 2), dtype=np.float32),
        rewards=np.array([
            0.34026529, 0.58263177, 0.84307509, 0.97651095, 0.81723901,
            0.22631398, 0.03421301, 0.97515046, 0.64311832, 0.65068933,
            0.17657714, 0.04783857, 0.73904013, 0.41364329, 0.52235551,
            0.24203526, 0.43328910
        ]),
        step_types=np.array(
            [StepType.FIRST] + [StepType.MID] * (lengths[0] - 2) +
            [StepType.TERMINAL] + [StepType.FIRST] + [StepType.MID] *
            (lengths[1] - 2) + [StepType.TERMINAL] + [StepType.FIRST] +
            [StepType.FIRST],
            dtype=StepType),
        env_infos={
            'success':
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                     dtype=bool)
        },
        agent_infos={},
        lengths=lengths)

    log_file = tempfile.NamedTemporaryFile()
    csv_output = dowel.CsvOutput(log_file.name)
    logger.add_output(csv_output)
    log_performance(7, batch, 0.8, prefix='test_log_performance')
    logger.log(tabular)
    logger.dump_output_type(dowel.CsvOutput)
    with open(log_file.name, 'r') as file:
        rows = list(csv.DictReader(file))
    res = {k: float(r) for (k, r) in rows[0].items()}
    assert res['test_log_performance/Iteration'] == 7
    assert res['test_log_performance/NumEpisodes'] == 4
    assert math.isclose(res['test_log_performance/SuccessRate'], 0.75)
    assert math.isclose(res['test_log_performance/TerminationRate'], 0.5)
    assert math.isclose(res['test_log_performance/AverageDiscountedReturn'],
                        1.1131040640673113)
    assert math.isclose(res['test_log_performance/AverageReturn'],
                        2.1659965525)
    assert math.isclose(res['test_log_performance/StdReturn'],
                        2.354067152038576)


@pytest.mark.serial
def test_log_multitask_performance_task_name():
    lengths = np.array([10, 5, 1, 1])
    batch = EpisodeBatch(
        EnvSpec(akro.Box(np.array([0., 0., 0.]), np.array([1., 1., 1.])),
                akro.Box(np.array([-1., -1.]), np.array([0., 0.]))),
        observations=np.ones((sum(lengths), 3), dtype=np.float32),
        last_observations=np.ones((len(lengths), 3), dtype=np.float32),
        actions=np.zeros((sum(lengths), 2), dtype=np.float32),
        rewards=np.array([
            0.34026529, 0.58263177, 0.84307509, 0.97651095, 0.81723901,
            0.22631398, 0.03421301, 0.97515046, 0.64311832, 0.65068933,
            0.17657714, 0.04783857, 0.73904013, 0.41364329, 0.52235551,
            0.24203526, 0.43328910
        ]),
        step_types=np.array([StepType.MID] * sum(lengths), dtype=StepType),
        env_infos={
            'success':
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                     dtype=bool),
            'task_name':
            np.array(['env1'] * 10 + ['env2'] * 5 + ['env1'] + ['env3'])
        },
        agent_infos={},
        lengths=lengths)

    log_file = tempfile.NamedTemporaryFile()
    csv_output = dowel.CsvOutput(log_file.name)
    logger.add_output(csv_output)
    log_multitask_performance(7, batch, 0.8)
    logger.log(tabular)
    logger.dump_output_type(dowel.CsvOutput)
    with open(log_file.name, 'r') as file:
        rows = list(csv.DictReader(file))
    res = {k: float(r) for (k, r) in rows[0].items()}
    assert res['env1/Iteration'] == 7
    assert res['env2/Iteration'] == 7
    assert res['env3/Iteration'] == 7
    assert res['env1/NumEpisodes'] == 2
    assert res['env2/NumEpisodes'] == 1
    assert res['env3/NumEpisodes'] == 1
    assert math.isclose(res['env1/SuccessRate'], 0.5)
    assert math.isclose(res['env2/SuccessRate'], 1.0)
    assert math.isclose(res['env3/SuccessRate'], 1.0)


@pytest.mark.serial
def test_log_multitask_performance_task_id():
    lengths = np.array([10, 5, 1, 1])
    batch = EpisodeBatch(
        EnvSpec(akro.Box(np.array([0., 0., 0.]), np.array([1., 1., 1.])),
                akro.Box(np.array([-1., -1.]), np.array([0., 0.]))),
        observations=np.ones((sum(lengths), 3), dtype=np.float32),
        last_observations=np.ones((len(lengths), 3), dtype=np.float32),
        actions=np.zeros((sum(lengths), 2), dtype=np.float32),
        rewards=np.array([
            0.34026529, 0.58263177, 0.84307509, 0.97651095, 0.81723901,
            0.22631398, 0.03421301, 0.97515046, 0.64311832, 0.65068933,
            0.17657714, 0.04783857, 0.73904013, 0.41364329, 0.52235551,
            0.24203526, 0.43328910
        ]),
        step_types=np.array([StepType.MID] * sum(lengths), dtype=StepType),
        env_infos={
            'success':
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                     dtype=bool),
            'task_id':
            np.array([1] * 10 + [3] * 5 + [1] + [4])
        },
        agent_infos={},
        lengths=lengths)

    log_file = tempfile.NamedTemporaryFile()
    csv_output = dowel.CsvOutput(log_file.name)
    logger.add_output(csv_output)
    log_multitask_performance(7, batch, 0.8, {
        1: 'env1',
        3: 'env2',
        4: 'env3',
        5: 'env4'
    })
    logger.log(tabular)
    logger.dump_output_type(dowel.CsvOutput)
    with open(log_file.name, 'r') as file:
        rows = list(csv.DictReader(file))
    res = {k: float(r) for (k, r) in rows[0].items()}
    assert res['env1/Iteration'] == 7
    assert res['env2/Iteration'] == 7
    assert res['env3/Iteration'] == 7
    assert res['env4/Iteration'] == 7
    assert res['env1/NumEpisodes'] == 2
    assert res['env2/NumEpisodes'] == 1
    assert res['env3/NumEpisodes'] == 1
    assert res['env4/NumEpisodes'] == 0
    assert math.isclose(res['env1/SuccessRate'], 0.5)
    assert math.isclose(res['env2/SuccessRate'], 1.0)
    assert math.isclose(res['env3/SuccessRate'], 1.0)
    assert math.isnan(res['env4/SuccessRate'])
    assert math.isnan(res['env4/AverageReturn'])


class TestOptimizerInterface(TfGraphTestCase):
    """Test class for tf & pytorch make_optimizer functions."""

    def test_tf_make_optimizer_with_type(self):
        """Test make_optimizer function with type as first argument."""
        optimizer_type = tf.compat.v1.train.AdamOptimizer
        lr = 0.123
        optimizer = make_optimizer(optimizer_type,
                                   learning_rate=lr,
                                   name='testOptimizer')
        assert isinstance(optimizer, optimizer_type)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        assert optimizer._name == 'testOptimizer'
        assert np.allclose(
            optimizer._lr, lr
        )  # Adam holds the value of learning rate in private variable self._lr

    def test_tf_make_optimizer_with_tuple(self):
        """Test make_optimizer function with tuple as first argument."""
        lr = 0.123
        optimizer_type = (tf.compat.v1.train.AdamOptimizer, {
            'learning_rate': lr
        })
        optimizer = make_optimizer(optimizer_type)
        # pylint: disable=isinstance-second-argument-not-valid-type
        assert isinstance(optimizer, optimizer_type)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        assert np.allclose(
            optimizer._lr, lr
        )  # Adam holds the value of learning rate in private variable self._lr

    def test_tf_make_optimizer_raise_value_error(self):
        """Test make_optimizer raises value error."""
        lr = 0.123
        optimizer_type = (tf.compat.v1.train.AdamOptimizer, {
            'learning_rate': lr
        })
        with pytest.raises(ValueError):
            _ = make_optimizer(optimizer_type, learning_rate=lr)

    def test_torch_make_optimizer_with_type(self):
        """Test make_optimizer function with type as first argument."""
        optimizer_type = torch.optim.Adam
        module = torch.nn.Linear(2, 1)
        lr = 0.123
        optimizer = make_optimizer(optimizer_type, module=module, lr=lr)
        assert isinstance(optimizer, optimizer_type)
        assert optimizer.defaults['lr'] == lr

    def test_torch_make_optimizer_with_tuple(self):
        """Test make_optimizer function with tuple as first argument."""
        optimizer_type = (torch.optim.Adam, {'lr': 0.1})
        module = torch.nn.Linear(2, 1)
        optimizer = make_optimizer(optimizer_type, module=module)
        # pylint: disable=isinstance-second-argument-not-valid-type
        assert isinstance(optimizer, optimizer_type)
        assert optimizer.defaults['lr'] == optimizer_type[1]['lr']

    def test_torch_make_optimizer_raise_value_error(self):
        """Test make_optimizer raises value error."""
        optimizer_type = (torch.optim.Adam, {'lr': 0.1})
        module = torch.nn.Linear(2, 1)
        with pytest.raises(ValueError):
            _ = make_optimizer(optimizer_type, module=module, lr=0.123)
