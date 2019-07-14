import numpy as np
import ray

from garage.envs.grid_world_env import GridWorldEnv
from garage.sampler import RaySampler, SamplerWorker
from garage.tf.envs import TfEnv
from garage.tf.samplers.on_policy_vectorized_sampler \
    import OnPolicyVectorizedSampler
from tests.fixtures.policies.ray_sampler_utils import MockAlgo, ScriptedPolicy


class TestSampler():
    def setup_method(self):
        ray.init(
            ignore_reinit_error=True,
            object_store_memory=10000000,
            redis_max_memory=10000000)

        self.env = TfEnv(GridWorldEnv(desc='4x4'))

    def teardown_method(self):
        self.env.close()
        ray.shutdown()

    def test_ray_batch_sampler(self):
        policy = ScriptedPolicy()
        algo = MockAlgo(self.env.spec, policy)
        sampler1 = RaySampler(
            algo,
            self.env,
            seed=100,
            num_processors=1,
            sampler_worker_cls=SamplerWorker)
        sampler1.start_worker()

        sampler2 = OnPolicyVectorizedSampler(algo, self.env)
        sampler2.start_worker()

        trajs1 = sampler1.obtain_samples(0, 16)[0]
        trajs2 = sampler2.obtain_samples(0, 1)[0]

        assert (trajs1['observations'].shape == np.array(
            trajs2['observations']).shape == (6, 16))
        traj2_action_shape = np.array(trajs2['actions']).shape
        assert (trajs1['actions'].shape == traj2_action_shape == (6, 4))
        assert (sum(trajs1['rewards']) == sum(trajs2['rewards']) == 1)
