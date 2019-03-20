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
                    name='input{}'.format(i))
                inputs.append(input_ph)
                policy_dist_info = maml_policy.wrapped_policy.dist_info_sym(
                    input_ph,
                    None,
                    name='dist_info_{}'.format(i),
                )
                action = policy_dist_info[
                    'mean'] + 0.1 * policy_dist_info['log_std']
                g_i = tf.gradients(action, policy_params)
                gradient_vars.append(g_i)

            outputs, _, _ = maml_policy.initialize(
                gradient_var=gradient_vars, inputs=inputs)

            # Check gradient from action_var to policy_params
            action_var = outputs[0][0]
            gradient = tf.gradients(action_var, policy_params)
            self.assertNotIn(None, gradient)
