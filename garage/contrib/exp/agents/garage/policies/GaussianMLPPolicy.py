class GaussianMLPPolicy:
    def __init__(
        self,
        env_spec,
        name="GaussianMLPPolicy",
        hidden_sizes=(32, 32),
        learn_std=True,
        init_std=1.0,
        adaptive_std=False,
        std_share_network=False,
        std_hidden_sizes=(32, 32),
        min_std=1e-6,
        # std_hidden_nonlinearity=tf.nn.tanh,
        # hidden_nonlinearity=tf.nn.tanh,
        output_nonlinearity=None,
        mean_network=None,
        std_network=None,
        std_parametrization='exp'):
        pass
