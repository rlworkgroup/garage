"""
Unit test for Bernoulli MLP Regressor with Model.

This test consists of two different regressors: bmr1, bmr2
bmr1 is a BernoulliMLPRegressor, which does not use
garage.tf.models while bmr2 does.

This test ensures the predictions and training performance from all
the regressors are the same, for the transition from using
BernoulliMLPRegressor to BernoulliMLPRegressorWithModel.

It covers predict and log_likelihood_sym.
"""
import numpy as np
import pytest
import tensorflow as tf

from garage.tf.regressors import BernoulliMLPRegressor
from garage.tf.regressors import BernoulliMLPRegressorWithModel
from tests.fixtures import TfGraphTestCase


@pytest.mark.parametrize('input_shape', [(1, ), (2, )])
@pytest.mark.parametrize('output_dim', [1, 2])
@pytest.mark.parametrize('use_trust_region', [True, False])
@pytest.mark.parametrize('normalize_inputs', [True, False])
class TestBernoulliMLPRegressorWithModelTransit(TfGraphTestCase):

    def test_all(self, input_shape, output_dim, use_trust_region,
                 normalize_inputs):
        n = 50
        xs = np.random.random((n, ) + input_shape)
        ys = np.random.randint(0, 1, (n, output_dim))

        bmr1 = BernoulliMLPRegressor(input_shape=input_shape,
                                     output_dim=output_dim,
                                     name='bmr1',
                                     use_trust_region=use_trust_region,
                                     normalize_inputs=normalize_inputs)
        bmr2 = BernoulliMLPRegressorWithModel(
            input_shape=input_shape,
            output_dim=output_dim,
            name='bmr2',
            use_trust_region=use_trust_region,
            normalize_inputs=normalize_inputs)

        self.sess.run(tf.compat.v1.global_variables_initializer())

        for a, b in zip(bmr1.get_params(), bmr2.get_params()):
            self.sess.run(a.assign(b))

        # Test predict()
        pred1 = bmr1.predict(xs)
        pred2 = bmr2.predict(xs)
        assert np.allclose(pred1, pred2)

        # Test log_likelihood_sym()
        new_xs_var2 = tf.compat.v1.placeholder(tf.float32, (n, ) + input_shape)
        new_ys_var2 = tf.compat.v1.placeholder(tf.float32,
                                               shape=(None, output_dim))

        likelihood1 = bmr1.predict_log_likelihood(xs, ys)

        likelihood2_out = bmr2.log_likelihood_sym(new_xs_var2,
                                                  new_ys_var2,
                                                  name='ll_sym')

        likelihood2 = self.sess.run(likelihood2_out,
                                    feed_dict={
                                        new_xs_var2: xs,
                                        new_ys_var2: ys
                                    })

        assert np.allclose(likelihood1, likelihood2)
