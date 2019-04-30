from garage.tf.algos.npo import NPO
from garage.tf.algos.npo import PGLoss
from garage.tf.optimizers import FirstOrderOptimizer


class VPG(NPO):
    """
    Vanilla Policy Gradient.
    """

    def __init__(self, optimizer=None, optimizer_args=None, **kwargs):
        if optimizer is None:
            default_args = dict(
                batch_size=None,
                max_epochs=1,
            )
            optimizer = FirstOrderOptimizer
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
        super(VPG, self).__init__(
            pg_loss=PGLoss.VANILLA,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            name='VPG',
            **kwargs)
