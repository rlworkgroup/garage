from unittest.mock import Mock

import numpy as np
import ray

from garage.envs.grid_world_env import GridWorldEnv
from garage.sampler import RaySampler, SamplerWorker
from garage.tf.envs import TfEnv
from garage.tf.samplers.on_policy_vectorized_sampler \
    import OnPolicyVectorizedSampler
from tests.fixtures.policies.ray_sampler_utils import ScriptedPolicy


class TestSampler():
    def setup_method(self):
        ray.init(local_mode=True, ignore_reinit_error=True)

        self.env = TfEnv(GridWorldEnv(desc='4x4'))
        self.policy = ScriptedPolicy()
        self.algo = Mock(
            env_spec=self.env.spec, policy=self.policy, max_path_length=16)

    def teardown_method(self):
        self.env.close()
        ray.shutdown()

    def test_ray_batch_sampler(self):
        sampler1 = RaySampler(
            self.algo,
            self.env,
            seed=100,
            num_processors=1,
            sampler_worker_cls=SamplerWorker)
        sampler1.start_worker()
        sampler2 = OnPolicyVectorizedSampler(self.algo, self.env)
        sampler2.start_worker()
        trajs1 = sampler1.obtain_samples(0, 16)
        trajs2 = sampler2.obtain_samples(0, 1)
        assert (trajs1[0]['observations'].shape == np.array(
            trajs2[0]['observations']).shape == (6, 16))
        traj2_action_shape = np.array(trajs2[0]['actions']).shape
        assert (trajs1[0]['actions'].shape == traj2_action_shape == (6, 4))
        assert (sum(trajs1[0]['rewards']) == sum(trajs2[0]['rewards']) == 1)

        true_obs = np.array(
            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.]])
        true_actions = np.array([[0., 0., 1., 0.], [0., 0., 1., 0.],
                                 [0., 1., 0., 0.], [0., 1., 0., 0.],
                                 [0., 1., 0., 0.], [0., 0., 1., 0.]])
        true_rewards = np.array([0, 0, 0, 0, 0, 1])
        for trajectory in trajs1:
            assert (np.array_equal(trajectory['observations'], true_obs))
            assert (np.array_equal(trajectory['actions'], true_actions))
            assert (np.array_equal(trajectory['rewards'], true_rewards))
