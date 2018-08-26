import tensorflow as tf

from garage.core import Serializable
from garage.misc import ext, logger
from garage.misc.overrides import overrides
from garage.tf.algos import BatchPolopt
from garage.tf.misc import tensor_utils
from garage.tf.optimizers import FirstOrderOptimizer


class VPG(BatchPolopt, Serializable):
    """
    Vanilla Policy Gradient.
    """

    def __init__(self,
                 env,
                 policy,
                 baseline,
                 optimizer=None,
                 optimizer_args=None,
                 name=None,
                 **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None
        self.name = name
        super(VPG, self).__init__(
            env=env, policy=policy, baseline=baseline, **kwargs)

    @overrides
    def init_opt(self):
        with tf.name_scope(self.name, "VPG"):
            is_recurrent = int(self.policy.recurrent)
            with tf.name_scope("inputs"):
                obs_var = self.env.observation_space.new_tensor_variable(
                    'obs',
                    extra_dims=1 + is_recurrent,
                )
                action_var = self.env.action_space.new_tensor_variable(
                    'action',
                    extra_dims=1 + is_recurrent,
                )
                advantage_var = tensor_utils.new_tensor(
                    name='advantage',
                    ndim=1 + is_recurrent,
                    dtype=tf.float32,
                )
                dist = self.policy.distribution

                old_dist_info_vars = {
                    k: tf.placeholder(
                        tf.float32,
                        shape=[None] * (1 + is_recurrent) + list(shape),
                        name='old_%s' % k)
                    for k, shape in dist.dist_info_specs
                }
                old_dist_info_vars_list = [
                    old_dist_info_vars[k] for k in dist.dist_info_keys
                ]

                state_info_vars = {
                    k: tf.placeholder(
                        tf.float32,
                        shape=[None] * (1 + is_recurrent) + list(shape),
                        name=k)
                    for k, shape in self.policy.state_info_specs
                }
                state_info_vars_list = [
                    state_info_vars[k] for k in self.policy.state_info_keys
                ]

                if is_recurrent:
                    valid_var = tf.placeholder(
                        tf.float32, shape=[None, None], name="valid")
                else:
                    valid_var = None

            dist_info_vars = self.policy.dist_info_sym(obs_var,
                                                       state_info_vars)
            logli = dist.log_likelihood_sym(action_var, dist_info_vars)
            kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

            # formulate as a minimization problem
            # The gradient of the surrogate objective is the policy gradient
            with tf.name_scope(
                    "surr_obj", values=[logli, advantage_var, valid_var]):
                if is_recurrent:
                    surr_obj = (
                        -tf.reduce_sum(logli * advantage_var * valid_var) /
                        tf.reduce_sum(valid_var))
                else:
                    surr_obj = -tf.reduce_mean(logli * advantage_var)
                tf.identity(surr_obj, name="surr_obj")

            with tf.name_scope("mean_kl", values=[kl, valid_var]):
                if is_recurrent:
                    mean_kl = tf.reduce_sum(
                        kl * valid_var) / tf.reduce_sum(valid_var)
                else:
                    mean_kl = tf.reduce_mean(kl)
                tf.identity(mean_kl, name="mean_kl")

            with tf.name_scope("max_kl", values=[kl, valid_var]):
                if is_recurrent:
                    max_kl = tf.reduce_max(kl * valid_var)
                else:
                    max_kl = tf.reduce_max(kl)
                tf.identity(max_kl, name="max_kl")

            input_list = [obs_var, action_var, advantage_var
                          ] + state_info_vars_list
            if is_recurrent:
                input_list.append(valid_var)

            self.optimizer.update_opt(
                loss=surr_obj, target=self.policy, inputs=input_list)

            f_kl = tensor_utils.compile_function(
                inputs=input_list + old_dist_info_vars_list,
                outputs=[mean_kl, max_kl],
            )
            self.opt_info = dict(f_kl=f_kl, )

    @overrides
    def optimize_policy(self, itr, samples_data):
        logger.log("optimizing policy")
        inputs = ext.extract(samples_data, "observations", "actions",
                             "advantages")
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        inputs += tuple(state_info_list)
        if self.policy.recurrent:
            inputs += (samples_data["valids"], )
        dist_info_list = [
            agent_infos[k] for k in self.policy.distribution.dist_info_keys
        ]
        loss_before = self.optimizer.loss(inputs)
        self.optimizer.optimize(inputs)
        loss_after = self.optimizer.loss(inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        mean_kl, max_kl = self.opt_info['f_kl'](*(
            list(inputs) + dist_info_list))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
