import csv
import tempfile

import cloudpickle
from dowel import CsvOutput, logger, tabular
import numpy as np
import pytest
import tensorflow as tf

from garage.envs import PointEnv
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

    def __init__(self, env, max_episode_length):
        self.env = env
        self.policy = RandomPolicy(self.env.spec.action_space)
        self.max_episode_length = max_episode_length

    def train(self, runner):
        del runner

    def get_exploration_policy(self):
        return self.policy

    def adapt_policy(self, exploration_policy, exploration_episodes):
        best_timestep = np.argmax(exploration_episodes.rewards)
        best_action = exploration_episodes.actions[best_timestep]
        return SingleActionPolicy(best_action)


@pytest.mark.serial
def test_meta_evaluator():
    set_seed(100)
    tasks = SetTaskSampler(PointEnv)
    max_episode_length = 200
    with tempfile.TemporaryDirectory() as log_dir_name:
        runner = LocalRunner(
            SnapshotConfig(snapshot_dir=log_dir_name,
                           snapshot_mode='last',
                           snapshot_gap=1))
        env = PointEnv()
        algo = OptimalActionInference(env=env,
                                      max_episode_length=max_episode_length)
        runner.setup(algo, env)
        meta_eval = MetaEvaluator(test_task_sampler=tasks,
                                  max_episode_length=max_episode_length,
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
        assert float(
            rows[0]['MetaTest/__unnamed_task__/TerminationRate']) < 1.0
        assert float(rows[0]['MetaTest/__unnamed_task__/Iteration']) == 0
        assert (float(rows[0]['MetaTest/__unnamed_task__/MaxReturn']) >= float(
            rows[0]['MetaTest/__unnamed_task__/AverageReturn']))
        assert (float(rows[0]['MetaTest/__unnamed_task__/AverageReturn']) >=
                float(rows[0]['MetaTest/__unnamed_task__/MinReturn']))
        assert float(rows[1]['MetaTest/__unnamed_task__/Iteration']) == 1


class MockAlgo:

    sampler_cls = LocalSampler

    def __init__(self, env, policy, max_episode_length, n_exploration_eps,
                 meta_eval):
        self.env = env
        self.policy = policy
        self.max_episode_length = max_episode_length
        self.n_exploration_eps = n_exploration_eps
        self.meta_eval = meta_eval

    def train(self, runner):
        for step in runner.step_epochs():
            if step % 5 == 0:
                self.meta_eval.evaluate(self)

    def get_exploration_policy(self):
        return self.policy

    def adapt_policy(self, exploration_policy, exploration_episodes):
        del exploration_policy
        assert len(exploration_episodes.lengths) == self.n_exploration_eps


def test_pickle_meta_evaluator():
    set_seed(100)
    tasks = SetTaskSampler(PointEnv)
    max_episode_length = 200
    env = PointEnv()
    n_eps = 3
    with tempfile.TemporaryDirectory() as log_dir_name:
        runner = LocalRunner(
            SnapshotConfig(snapshot_dir=log_dir_name,
                           snapshot_mode='last',
                           snapshot_gap=1))
        meta_eval = MetaEvaluator(test_task_sampler=tasks,
                                  max_episode_length=max_episode_length,
                                  n_test_tasks=10,
                                  n_exploration_eps=n_eps)
        policy = RandomPolicy(env.spec.action_space)
        algo = MockAlgo(env, policy, max_episode_length, n_eps, meta_eval)
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
    tasks = SetTaskSampler(PointEnv)
    max_episode_length = 200
    env = PointEnv()
    n_eps = 3
    with tempfile.TemporaryDirectory() as log_dir_name:
        ctxt = SnapshotConfig(snapshot_dir=log_dir_name,
                              snapshot_mode='none',
                              snapshot_gap=1)
        with LocalTFRunner(ctxt) as runner:
            meta_eval = MetaEvaluator(test_task_sampler=tasks,
                                      max_episode_length=max_episode_length,
                                      n_test_tasks=10,
                                      n_exploration_eps=n_eps)
            policy = GaussianMLPPolicy(env.spec)
            algo = MockAlgo(env, policy, max_episode_length, n_eps, meta_eval)
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
