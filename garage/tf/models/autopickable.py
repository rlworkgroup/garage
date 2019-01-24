"""
Autopickable.
"""
import tensorflow as tf

# flake8: noqa


class AutoPickable:
    def __getstate__(self):
        _args = []
        # if it is a model
        state = self.__dict__.copy()

        for k, v in self.__dict__.items():
            if isinstance(v, tf.Tensor):
                _args.append(k)
                del state[k]
        del state['model']
        state['model'] = self.model.get_config()
        # pdb.set_trace()

        # state['weights'] = self.model.get_weights()
        # pdb.set_trace()

        # state['model_field'] = _args
        # pdb.set_trace()
        del state['_dist']
        return state

    def __setstate__(self, d):
        # if it is a model
        model = tf.keras.models.model_from_json(d['model'])
        model.set_weights(d['weights'])
        self.model = model
        for i, k in enumerate(d['args']):
            if k != '_input':
                setattr(self, k, model.outputs[i])
