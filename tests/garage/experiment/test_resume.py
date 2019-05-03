import shutil
import tempfile

from garage.experiment import LocalRunner
from garage.logger import snapshotter
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import VPG
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from tests.fixtures import TfGraphTestCase


def fixture_exp():
    with LocalRunner() as runner:
        env = TfEnv(env_name='CartPole-v1')

        policy = CategoricalMLPPolicy(
            name='policy', env_spec=env.spec, hidden_sizes=(8, 8))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = VPG(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            optimizer_args=dict(tf_optimizer_args=dict(learning_rate=0.01, )))

        runner.setup(algo, env)
        runner.train(n_epochs=5, batch_size=100)


class TestResume(TfGraphTestCase):
    def test_resume(self):
        # Manually create and remove temp folder
        # Otherwise, tempfile unexpectedly removes folder in child folder
        folder = tempfile.mkdtemp()
        snapshotter.snapshot_dir = folder
        snapshotter.snapshot_mode = 'last'

        fixture_exp()
        self.tearDown()
        self.setUp()

        with LocalRunner() as runner:
            args = runner.restore(folder, resume_now=False)
            assert args['n_epochs'] == 5, \
                'Snapshot should save training parameters'
            assert args['start_epoch'] == 5, \
                'Last experiment should end at 5th iterations'
            args['n_epochs'] = 10
            runner.train(**args)

        shutil.rmtree(folder)
