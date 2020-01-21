import csv
import tempfile

from dowel import CsvOutput, logger, tabular
import numpy as np
import pytest

from garage.envs import GarageEnv, PointEnv
from garage.experiment import SnapshotConfig
from garage.experiment.deterministic import set_seed
from garage.experiment.local_runner import LocalRunner
from garage.experiment.meta_evaluator import MetaEvaluator
from garage.experiment.task_sampler import SetTaskSampler
from garage.np.algos import MetaRLAlgorithm
from garage.sampler import LocalSampler


class RandomPolicy:

    def __init__(self, action_space):
        self._action_space = action_space

    def reset(self):
        pass

    def get_action(self, observation):
        del observation
        return self._action_space.sample(), {}


class SingleActionPolicy:

    def __init__(self, action):
        self._action = action

    def reset(self):
        pass

    def get_action(self, observation):
        del observation
        return self._action, {}


class OptimalActionInference(MetaRLAlgorithm):

    sampler_cls = LocalSampler

    def __init__(self, env, max_path_length):
        self.env = env
        self.policy = RandomPolicy(self.env.spec.action_space)
        self.max_path_length = max_path_length

    def train(self, runner):
        del runner

    def get_exploration_policy(self):
        return self.policy

    def adapt_policy(self, exploration_policy, exploration_trajectories):
        best_timestep = np.argmax(exploration_trajectories.rewards)
        best_action = exploration_trajectories.actions[best_timestep]
        return SingleActionPolicy(best_action)


@pytest.mark.serial
def test_meta_evaluator():
    set_seed(100)
    tasks = SetTaskSampler(PointEnv)
    max_path_length = 200
    with tempfile.TemporaryDirectory() as log_dir_name:
        runner = LocalRunner(
            SnapshotConfig(snapshot_dir=log_dir_name,
                           snapshot_mode='last',
                           snapshot_gap=1))
        env = GarageEnv(PointEnv())
        algo = OptimalActionInference(env=env, max_path_length=max_path_length)
        runner.setup(algo, env)
        meta_eval = MetaEvaluator(runner,
                                  test_task_sampler=tasks,
                                  max_path_length=max_path_length,
                                  n_test_tasks=10)
        log_file = tempfile.NamedTemporaryFile()
        csv_output = CsvOutput(log_file.name)
        logger.add_output(csv_output)
        meta_eval.evaluate(algo)
        logger.log(tabular)
        meta_eval.evaluate(algo)
        logger.log(tabular)
        logger.dump_output_type(CsvOutput)
        logger.remove_output_type(CsvOutput)
        with open(log_file.name, 'r') as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 2
        assert float(rows[0]['MetaTest/CompletionRate']) < 1.0
        assert float(rows[0]['MetaTest/Iteration']) == 0
        assert (float(rows[0]['MetaTest/MaxReturn']) >= float(
            rows[0]['MetaTest/AverageReturn']))
        assert (float(rows[0]['MetaTest/AverageReturn']) >= float(
            rows[0]['MetaTest/MinReturn']))
        assert float(rows[1]['MetaTest/Iteration']) == 1
