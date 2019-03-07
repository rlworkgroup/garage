import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies.gaussian_mlp_policy_with_model import GaussianMLPPolicyWithModel
from garage.tf.policies import MamlPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestMamlPolicies(TfGraphTestCase):
    def test_maml_policy(self):
        box_env = TfEnv(DummyBoxEnv())
        with self.sess.as_default():
            policy = GaussianMLPPolicyWithModel(
                env_spec=box_env, hidden_sizes=(1, ))

            maml_policy = MamlPolicy(wrapped_policy=policy, n_tasks=2)

            gradient_vars = list()
            inputs = list()
            policy_params = policy.get_params()
            policy_inputs = policy.model.networks['default'].input

            for i in range(2):
                input_ph = tf.placeholder(
                    shape=policy_inputs.shape,
                    dtype=policy_inputs.dtype,
                    name="task_input")
                inputs.append(input_ph)

                g_i = list()
                for p in policy_params:
                    grad = tf.placeholder(
                        dtype=p.dtype,
                        shape=p.shape,
                        name="maml_grad/task{}/{}".format(i, p.name[:-2]))
                    g_i.append(grad)
                gradient_vars.append(g_i)

            maml_policy.initialize(gradient_var=gradient_vars, inputs=inputs)
