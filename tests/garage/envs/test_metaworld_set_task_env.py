import pytest

from garage.envs import MetaWorldSetTaskEnv
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.sampler import LocalSampler, SetTaskUpdate, WorkerFactory


class RandomPolicy:

    def __init__(self, action_space):
        self._action_space = action_space

    def reset(self):
        pass

    def get_action(self, observation):
        del observation
        return self._action_space.sample(), {}


@pytest.mark.mujoco
def test_sample_and_step():
    # Import, construct environments here to avoid using up too much
    # resources if this test isn't run.
    # pylint: disable=import-outside-toplevel
    import metaworld
    ml1 = metaworld.ML1('push-v1')
    env = MetaWorldSetTaskEnv(ml1, 'train')
    assert env.num_tasks == 50
    task = env.sample_tasks(1)[0]
    env.set_task(task)
    env.step(env.action_space.sample())
    env.close()
    env2 = MetaWorldSetTaskEnv()
    env2.set_task(task)
    env2.step(env.action_space.sample())
    env2.close()
    tasks = env.sample_tasks(100)
    assert len(tasks) == 100


@pytest.mark.mujoco
def test_forbidden_cases():
    # Import, construct environments here to avoid using up too much
    # resources if this test isn't run.
    # pylint: disable=import-outside-toplevel
    import metaworld
    ml1 = metaworld.ML1('push-v1')
    with pytest.raises(ValueError):
        MetaWorldSetTaskEnv(ml1, 'train', add_env_onehot=True)
    with pytest.raises(ValueError):
        MetaWorldSetTaskEnv(ml1, 'Test')


@pytest.mark.mujoco
def test_onehots_consistent_with_task_sampler():
    # Import, construct environments here to avoid using up too much
    # resources if this test isn't run.
    # pylint: disable=import-outside-toplevel
    import metaworld
    mt10 = metaworld.MT10()
    env = MetaWorldSetTaskEnv(mt10, 'train', add_env_onehot=True)
    policy = RandomPolicy(env.action_space)
    workers = WorkerFactory(seed=100, max_episode_length=1, n_workers=10)
    sampler1 = LocalSampler.from_worker_factory(workers, policy, env)
    env_ups = [
        SetTaskUpdate(MetaWorldSetTaskEnv, task, None)
        for task in env.sample_tasks(10)
    ]
    samples1 = sampler1.obtain_exact_episodes(1, policy, env_ups)
    task_sampler = MetaWorldTaskSampler(mt10, 'train', add_env_onehot=True)
    env_ups = task_sampler.sample(10)
    sampler2 = LocalSampler.from_worker_factory(workers, policy, env_ups)
    samples2 = sampler2.obtain_exact_episodes(1, policy, env_ups)
    name_to_obs1 = {}
    for obs1, name1 in zip(samples1.observations,
                           samples1.env_infos['task_name']):
        name_to_obs1[name1] = obs1
    for obs2, name2 in zip(samples2.observations,
                           samples2.env_infos['task_name']):
        assert (name_to_obs1[name2][-10:] == obs2[-10:]).all()
