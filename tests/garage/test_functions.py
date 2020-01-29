"""Test root level functions in garage."""
import csv
import math
import tempfile

import akro
import dowel
from dowel import logger, tabular
import numpy as np
import pytest

from garage import log_performance, TrajectoryBatch
from garage.envs import EnvSpec


@pytest.mark.serial
def test_log_performance():
    lengths = np.array([10, 5, 1, 1])
    batch = TrajectoryBatch(
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
        terminals=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
                           dtype=bool),
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
    assert res['test_log_performance/NumTrajs'] == 4
    assert math.isclose(res['test_log_performance/SuccessRate'], 0.75)
    assert math.isclose(res['test_log_performance/CompletionRate'], 0.5)
    assert math.isclose(res['test_log_performance/AverageDiscountedReturn'],
                        1.1131040640673113)
    assert math.isclose(res['test_log_performance/AverageReturn'],
                        2.1659965525)
    assert math.isclose(res['test_log_performance/StdReturn'],
                        2.354067152038576)
