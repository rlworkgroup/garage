# yapf: disable
import numpy as np
import ray

from garage import TimeStepBatch
from garage.envs import PointEnv
from garage.experiment import deterministic, LocalRunner
from garage.sampler import LocalSampler, WorkerFactory
from garage.torch.algos import BC
from garage.torch.policies import (DeterministicMLPPolicy,
                                   GaussianMLPPolicy,
                                   Policy)

from tests.fixtures import snapshot_config
from tests.fixtures.sampler import ray_local_session_fixture  # NOQA

# yapf: enable


class OptimalPolicy(Policy):
    """Optimal policy for PointEnv.

    Args:
        env_spec (EnvSpec): The environment spec.
        goal (np.ndarray): The goal location of the environment.

    """

    # No forward method
    # pylint: disable=abstract-method

    def __init__(self, env_spec, goal):
        super().__init__(env_spec, 'OptimalPolicy')
        self.goal = goal

    def get_action(self, observation):
        """Get action given observation.

        Args:
            observation (np.ndarray): Observation from PointEnv. Should have
                length at least 2.

        Returns:
            tuple:
                * np.ndarray: Optimal action in the environment. Has length 2.
                * dict[str, np.ndarray]: Agent info (empty).

        """
        return self.goal - observation[:2], {}

    def get_actions(self, observations):
        """Get actions given observations.

        Args:
            observations (np.ndarray): Observations from the environment.
                Has shape :math:`(B, O)`, where :math:`B` is the batch
                dimension and :math:`O` is the observation dimensionality (at
                least 2).

        Returns:
            tuple:
                * np.ndarray: Batch of optimal actions.
                    Has shape :math:`(B, 2)`, where :math:`B` is the batch
                    dimension.
                Optimal action in the environment.
                * dict[str, np.ndarray]: Agent info (empty).

        """
        return (self.goal[np.newaxis, :].repeat(len(observations), axis=0) -
                observations[:, :2]), {}


def run_bc(runner, algo, batch_size):
    # Don't uninitialize ray after calling `train`.
    runner._shutdown_worker = lambda: None
    runner.train(0, batch_size=batch_size)
    initial_loss = np.mean(algo._train_once(runner, 0))
    runner.train(2, batch_size=batch_size)
    final_loss = np.mean(algo._train_once(runner, 3))
    assert final_loss < initial_loss


def test_bc_point_deterministic(ray_local_session_fixture):  # NOQA
    del ray_local_session_fixture
    assert ray.is_initialized()
    deterministic.set_seed(100)
    runner = LocalRunner(snapshot_config)
    goal = np.array([1., 1.])
    env = PointEnv(goal=goal)
    expert = OptimalPolicy(env.spec, goal=goal)
    policy = DeterministicMLPPolicy(env.spec, hidden_sizes=[8, 8])
    batch_size = 600
    algo = BC(env.spec,
              policy,
              batch_size=batch_size,
              source=expert,
              max_episode_length=200,
              policy_lr=1e-2,
              loss='mse')
    runner.setup(algo, env)
    run_bc(runner, algo, batch_size)


def test_bc_point(ray_local_session_fixture):  # NOQA
    del ray_local_session_fixture
    assert ray.is_initialized()
    deterministic.set_seed(100)
    runner = LocalRunner(snapshot_config)
    goal = np.array([1., 1.])
    env = PointEnv(goal=goal)
    expert = OptimalPolicy(env.spec, goal=goal)
    policy = GaussianMLPPolicy(env.spec, [4])
    batch_size = 400
    algo = BC(env.spec,
              policy,
              batch_size=batch_size,
              source=expert,
              max_episode_length=200,
              policy_lr=1e-2,
              loss='log_prob')
    runner.setup(algo, env)
    run_bc(runner, algo, batch_size)


def expert_source(env, goal, max_episode_length, n_eps):
    expert = OptimalPolicy(env.spec, goal=goal)
    workers = WorkerFactory(seed=100, max_episode_length=max_episode_length)
    expert_sampler = LocalSampler.from_worker_factory(workers, expert, env)
    for _ in range(n_eps):
        eps_batch = expert_sampler.obtain_samples(0, max_episode_length, None)
        yield TimeStepBatch.from_episode_batch(eps_batch)


def test_bc_point_sample_batches():
    deterministic.set_seed(100)
    runner = LocalRunner(snapshot_config)
    goal = np.array([1., 1.])
    env = PointEnv(goal=goal)
    max_episode_length = 200
    source = list(expert_source(env, goal, max_episode_length, 5))
    policy = DeterministicMLPPolicy(env.spec, hidden_sizes=[8, 8])
    batch_size = 600
    algo = BC(env.spec,
              policy,
              batch_size=batch_size,
              source=source,
              policy_lr=1e-2,
              loss='mse')
    runner.setup(algo, env)
    run_bc(runner, algo, batch_size)
