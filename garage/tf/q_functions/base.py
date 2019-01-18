import tensorflow as tf

from garage.tf.core import Parameterized


class QFunction(Parameterized):
    def build_net(self, name):
        raise NotImplementedError

    def get_qval_sym(self, input_phs):
        raise NotImplementedError

    def log_diagnostics(self, paths):
        pass

    def get_trainable_vars(self, scope=None):
        scope = scope if scope else self.name
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def get_global_vars(self, scope=None):
        scope = scope if scope else self.name
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def get_regularizable_vars(self, scope=None):
        scope = scope if scope else self.name
        reg_vars = [
            var for var in self.get_trainable_vars(scope=scope)
            if 'W' in var.name and 'output' not in var.name
        ]
        return reg_vars
