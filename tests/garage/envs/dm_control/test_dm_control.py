import unittest

from dm_control import suite

from garage.envs import normalize
from garage.envs.dm_control import DmControlEnv


def run_task(domain_name, task_name):
    print("run: domain %s task %s" % (domain_name, task_name))
    dm_control_env = normalize(
        DmControlEnv(
            domain_name=domain_name,
            task_name=task_name,
            plot=True,
            width=600,
            height=400),
        normalize_obs=False,
        normalize_reward=False)

    time_step = dm_control_env.reset()
    action_spec = dm_control_env.action_space
    for _ in range(5):
        dm_control_env.render()
        action = action_spec.sample()
        next_obs, reward, done, info = dm_control_env.step(action)
        if done:
            break

    dm_control_env.close()


class TestDmControl(unittest.TestCase):
    def test_dm_control(self):
        for domain, task in suite.ALL_TASKS:
            run_task(domain, task)

        print("Congratulation! All tasks are done!")
