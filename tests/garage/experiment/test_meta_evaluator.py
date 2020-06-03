import csv
import tempfile

import cloudpickle
from dowel import CsvOutput, logger, tabular
import numpy as np
import pytest
import tensorflow as tf

from garage.envs import GarageEnv, PointEnv
from garage.experiment import LocalTFRunner, MetaEvaluator, SnapshotConfig
from garage.experiment.deterministic import set_seed
from garage.experiment.local_runner import LocalRunner
from garage.experiment.task_sampler import SetTaskSampler
from garage.np.algos import MetaRLAlgorithm
from garage.sampler import LocalSampler
from garage.tf.policies import GaussianMLPPolicy


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
    tasks = SetTaskSampler(lambda: GarageEnv(PointEnv()))
    max_path_length = 200
    with tempfile.TemporaryDirectory() as log_dir_name:
        runner = LocalRunner(
            SnapshotConfig(snapshot_dir=log_dir_name,
                           snapshot_mode='last',
                           snapshot_gap=1))
        env = GarageEnv(PointEnv())
        algo = OptimalActionInference(env=env, max_path_length=max_path_length)
        runner.setup(algo, env)
        meta_eval = MetaEvaluator(test_task_sampler=tasks,
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
        assert float(rows[0]['MetaTest/__unnamed_task__/CompletionRate']) < 1.0
        assert float(rows[0]['MetaTest/__unnamed_task__/Iteration']) == 0
        assert (float(rows[0]['MetaTest/__unnamed_task__/MaxReturn']) >= float(
            rows[0]['MetaTest/__unnamed_task__/AverageReturn']))
        assert (float(rows[0]['MetaTest/__unnamed_task__/AverageReturn']) >=
                float(rows[0]['MetaTest/__unnamed_task__/MinReturn']))
        assert float(rows[1]['MetaTest/__unnamed_task__/Iteration']) == 1


class MockAlgo:

    sampler_cls = LocalSampler

    def __init__(self, env, policy, max_path_length, n_exploration_traj,
                 meta_eval):
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.n_exploration_traj = n_exploration_traj
        self.meta_eval = meta_eval

    def train(self, runner):
        for step in runner.step_epochs():
            if step % 5 == 0:
                self.meta_eval.evaluate(self)

    def get_exploration_policy(self):
        return self.policy

    def adapt_policy(self, exploration_policy, exploration_trajectories):
        del exploration_policy
        assert (len(
            exploration_trajectories.lengths) == self.n_exploration_traj)


class MockTFAlgo(MockAlgo):

    sampler_cls = LocalSampler

    def __init__(self, env, policy, max_path_length, n_exploration_traj,
                 meta_eval):
        super().__init__(env, policy, max_path_length, n_exploration_traj,
                         meta_eval)
        self._build()

    def _build(self):
        input_var = tf.compat.v1.placeholder(
            tf.float32,
            shape=(None, None, self.env.observation_space.flat_dim))
        self.policy.build(input_var)

    def __setstate__(self, state):
        """Parameters to restore from snapshot.

        Args:
            state (dict): Parameters to restore from.

        """
        self.__dict__ = state
        self._build()


def test_pickle_meta_evaluator():
    set_seed(100)
    tasks = SetTaskSampler(lambda: GarageEnv(PointEnv()))
    max_path_length = 200
    env = GarageEnv(PointEnv())
    n_traj = 3
    with tempfile.TemporaryDirectory() as log_dir_name:
        runner = LocalRunner(
            SnapshotConfig(snapshot_dir=log_dir_name,
                           snapshot_mode='last',
                           snapshot_gap=1))
        meta_eval = MetaEvaluator(test_task_sampler=tasks,
                                  max_path_length=max_path_length,
                                  n_test_tasks=10,
                                  n_exploration_traj=n_traj)
        policy = RandomPolicy(env.spec.action_space)
        algo = MockAlgo(env, policy, max_path_length, n_traj, meta_eval)
        runner.setup(algo, env)
        log_file = tempfile.NamedTemporaryFile()
        csv_output = CsvOutput(log_file.name)
        logger.add_output(csv_output)
        meta_eval.evaluate(algo)
        meta_eval_pickle = cloudpickle.dumps(meta_eval)
        meta_eval2 = cloudpickle.loads(meta_eval_pickle)
        meta_eval2.evaluate(algo)


def test_meta_evaluator_with_tf():
    set_seed(100)
    tasks = SetTaskSampler(lambda: GarageEnv(PointEnv()))
    max_path_length = 200
    env = GarageEnv(PointEnv())
    n_traj = 3
    with tempfile.TemporaryDirectory() as log_dir_name:
        ctxt = SnapshotConfig(snapshot_dir=log_dir_name,
                              snapshot_mode='none',
                              snapshot_gap=1)
        with LocalTFRunner(ctxt) as runner:
            meta_eval = MetaEvaluator(test_task_sampler=tasks,
                                      max_path_length=max_path_length,
                                      n_test_tasks=10,
                                      n_exploration_traj=n_traj)
            policy = GaussianMLPPolicy(env.spec)
            algo = MockTFAlgo(env, policy, max_path_length, n_traj, meta_eval)
            runner.setup(algo, env)
            log_file = tempfile.NamedTemporaryFile()
            csv_output = CsvOutput(log_file.name)
            logger.add_output(csv_output)
            meta_eval.evaluate(algo)
            algo_pickle = cloudpickle.dumps(algo)
        tf.compat.v1.reset_default_graph()
        with LocalTFRunner(ctxt) as runner:
            algo2 = cloudpickle.loads(algo_pickle)
            runner.setup(algo2, env)
            runner.train(10, 0)
