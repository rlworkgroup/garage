"""Tests for garage.tf.optimizers.conjugateGradientOptimizer"""
import pickle

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.optimizers.conjugate_gradient_optimizer import (
    cg, ConjugateGradientOptimizer, FiniteDifferenceHvp, PearlmutterHvp)
from garage.tf.policies import Policy
from tests.fixtures import TfGraphTestCase


class HelperPolicy(Policy):
    """Helper policy class for testing hvp classes"""

    def __init__(self, n_vars, name='OneParamPolicy'):
        super().__init__(name, None)
        with tf.compat.v1.variable_scope(self.name) as vs:
            self._variable_scope = vs
            _ = [tf.Variable([0.]) for _ in range(n_vars)]

    def get_action(self, observation):
        pass

    def get_actions(self, observations):
        pass


class TestConjugateGradientOptimizer(TfGraphTestCase):
    """Test class for ConjugateGradientOptimizer and HVP classes"""

    def test_cg(self):
        """Solve Ax = b using Conjugate gradient method."""
        a = np.linspace(-np.pi, np.pi, 25).reshape((5, 5))
        a = a.T.dot(a)  # make sure a is positive semi-definite
        b = np.linspace(-np.pi, np.pi, 5)
        x = cg(a.dot, b, cg_iters=5)
        assert np.allclose(a.dot(x), b)

    def test_pickleable(self):
        policy = HelperPolicy(n_vars=1)
        x = policy.get_params()[0]
        a_val = np.array([5.0], dtype=np.float32)
        a = tf.constant(a_val)
        loss = a * (x**2)
        constraint = (loss, 0.0)
        self.sess.run(tf.global_variables_initializer())
        opt = ConjugateGradientOptimizer()
        opt.update_opt(loss, policy, constraint, [a])
        opt.optimize([a_val])
        loss_before = opt.loss([a_val])
        opt = pickle.loads(pickle.dumps(opt))
        opt.update_opt(loss, policy, constraint, [a])
        loss_after = opt.loss([a_val])
        assert np.equal(loss_before, loss_after)


class TestPearlmutterHvp(TfGraphTestCase):
    """Test class for PearlmutterHvp"""

    def test_pearl_mutter_hvp_1x1(self):
        """Test Hessian-vector product for a function with one variable."""
        policy = HelperPolicy(n_vars=1)
        x = policy.get_params()[0]
        a_val = np.array([5.0])
        a = tf.constant([0.0])
        f = a * (x**2)
        expected_hessian = 2 * a_val
        vector = np.array([10.0])
        expected_hvp = expected_hessian * vector
        reg_coeff = 1e-5
        hvp = PearlmutterHvp()

        self.sess.run(tf.global_variables_initializer())
        hvp.update_hvp(f, policy, (a, ), reg_coeff)
        hx = hvp.build_eval(np.array([a_val]))
        computed_hvp = hx(vector)
        np.allclose(computed_hvp, expected_hvp)

    @pytest.mark.parametrize('a_val, b_val, x_val, y_val, vector', [
        (1.0, 1.0, 1.0, 1.0, [10.0, 20.0]),
        (5.0, 10.0, -2.0, 5.0, [0.0, -1.0]),
        (0.0, 0.0, 1.1, 0.02, [0.0, 0.0]),
        (-2.2, -1.5, -12.3, 34.8, [2.2, 5.3]),
        (-1.5, 0.0, -0.002, 4.93, [0.1, -0.02]),
    ])
    def test_pearl_mutter_hvp_2x2(self, a_val, b_val, x_val, y_val, vector):
        """Test Hessian-vector product for a function with two variables."""
        a_val = [a_val]
        b_val = [b_val]
        vector = np.array([vector], dtype=np.float32)

        policy = HelperPolicy(n_vars=2)
        params = policy.get_params()
        x, y = params[0], params[1]
        a = tf.constant(a_val)
        b = tf.constant(b_val)
        f = a * (x**2) + b * (y**2)

        expected_hessian = compute_hessian(f, [x, y])
        expected_hvp = tf.matmul(vector, expected_hessian)
        reg_coeff = 1e-5
        hvp = PearlmutterHvp()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(x.assign([x_val]))
        self.sess.run(y.assign([y_val]))
        hvp.update_hvp(f, policy, (a, b), reg_coeff)
        hx = hvp.build_eval((np.array(a_val), np.array(b_val)))
        hvp = hx(vector[0])
        expected_hvp = expected_hvp.eval()
        np.allclose(hvp, expected_hvp)

    @pytest.mark.parametrize('a_val, b_val, x_val, y_val, vector', [
        (1.0, 1.0, 1.0, 1.0, [10.0, 20.0]),
        (5.0, 10.0, -2.0, 5.0, [0.0, -1.0]),
        (0.0, 0.0, 1.1, 0.02, [0.0, 0.0]),
        (-2.2, -1.5, -12.3, 34.8, [2.2, 5.3]),
        (-1.5, 0.0, -0.002, 4.93, [0.1, -0.02]),
    ])
    def test_pearl_mutter_hvp_2x2_non_diagonal(self, a_val, b_val, x_val,
                                               y_val, vector):
        """Test Hessian-vector product for a function with two variables whose Hessian
        is non-diagonal.
        """
        a_val = [a_val]
        b_val = [b_val]
        vector = np.array([vector], dtype=np.float32)

        policy = HelperPolicy(n_vars=2)
        params = policy.get_params()
        x, y = params[0], params[1]
        a = tf.constant(a_val)
        b = tf.constant(b_val)
        f = a * (x**3) + b * (y**3) + (x**2) * y + (y**2) * x

        expected_hessian = compute_hessian(f, [x, y])
        expected_hvp = tf.matmul(vector, expected_hessian)
        reg_coeff = 1e-5
        hvp = PearlmutterHvp()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(x.assign([x_val]))
        self.sess.run(y.assign([y_val]))
        hvp.update_hvp(f, policy, (a, b), reg_coeff)
        hx = hvp.build_eval((np.array(a_val), np.array(b_val)))
        hvp = hx(vector[0])
        expected_hvp = expected_hvp.eval()
        np.allclose(hvp, expected_hvp)

    def test_pickleable(self):
        policy = HelperPolicy(n_vars=1)
        x = policy.get_params()[0]
        a_val = np.array([5.0])
        a = tf.constant([0.0])
        f = a * (x**2)
        vector = np.array([10.0])
        reg_coeff = 1e-5
        hvp = PearlmutterHvp()

        self.sess.run(tf.global_variables_initializer())
        hvp.update_hvp(f, policy, (a, ), reg_coeff)
        hx = hvp.build_eval(np.array([a_val]))
        before_pickle = hx(vector)

        hvp = pickle.loads(pickle.dumps(hvp))
        hvp.update_hvp(f, policy, (a, ), reg_coeff)
        after_pickle = hx(vector)
        assert np.equal(before_pickle, after_pickle)


class TestFiniteDifferenceHvp(TfGraphTestCase):
    """Test class for FiniteDifferenceHvp"""

    def test_finite_difference_hvp(self):
        """Test Hessian-vector product for a function with one variable."""
        policy = HelperPolicy(n_vars=1)
        x = policy.get_params()[0]
        a_val = np.array([5.0])
        a = tf.constant([0.0])
        f = a * (x**2)
        expected_hessian = 2 * a_val
        vector = np.array([10.0])
        expected_hvp = expected_hessian * vector
        reg_coeff = 1e-5
        hvp = FiniteDifferenceHvp()

        self.sess.run(tf.global_variables_initializer())
        hvp.update_hvp(f, policy, (a, ), reg_coeff)
        hx = hvp.build_eval(np.array([a_val]))
        computed_hvp = hx(vector)
        np.allclose(computed_hvp, expected_hvp)

    @pytest.mark.parametrize('a_val, b_val, x_val, y_val, vector', [
        (1.0, 1.0, 1.0, 1.0, [10.0, 20.0]),
        (5.0, 10.0, -2.0, 5.0, [0.0, -1.0]),
        (0.0, 0.0, 1.1, 0.02, [0.0, 0.0]),
        (-2.2, -1.5, -12.3, 34.8, [2.2, 5.3]),
        (-1.5, 0.0, -0.002, 4.93, [0.1, -0.02]),
    ])
    def test_finite_difference_hvp_2x2(self, a_val, b_val, x_val, y_val,
                                       vector):
        """Test Hessian-vector product for a function with two variables."""
        a_val = [a_val]
        b_val = [b_val]
        vector = np.array([vector], dtype=np.float32)

        policy = HelperPolicy(n_vars=2)
        params = policy.get_params()
        x, y = params[0], params[1]
        a = tf.constant(a_val)
        b = tf.constant(b_val)
        f = a * (x**2) + b * (y**2)

        expected_hessian = compute_hessian(f, [x, y])
        expected_hvp = tf.matmul(vector, expected_hessian)
        reg_coeff = 1e-5
        hvp = FiniteDifferenceHvp()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(x.assign([x_val]))
        self.sess.run(y.assign([y_val]))
        hvp.update_hvp(f, policy, (a, b), reg_coeff)
        hx = hvp.build_eval((np.array(a_val), np.array(b_val)))
        hvp = hx(vector[0])
        expected_hvp = expected_hvp.eval()
        np.allclose(hvp, expected_hvp)

    @pytest.mark.parametrize('a_val, b_val, x_val, y_val, vector', [
        (1.0, 1.0, 1.0, 1.0, [10.0, 20.0]),
        (5.0, 10.0, -2.0, 5.0, [0.0, -1.0]),
        (0.0, 0.0, 1.1, 0.02, [0.0, 0.0]),
        (-2.2, -1.5, -12.3, 34.8, [2.2, 5.3]),
        (-1.5, 0.0, -0.002, 4.93, [0.1, -0.02]),
    ])
    def test_finite_difference_hvp_2x2_non_diagonal(self, a_val, b_val, x_val,
                                                    y_val, vector):
        """Test Hessian-vector product for a function with two variables whose Hessian
        is non-diagonal.
        """
        a_val = [a_val]
        b_val = [b_val]
        vector = np.array([vector], dtype=np.float32)

        policy = HelperPolicy(n_vars=2)
        params = policy.get_params()
        x, y = params[0], params[1]
        a = tf.constant(a_val)
        b = tf.constant(b_val)
        f = a * (x**3) + b * (y**3) + (x**2) * y + (y**2) * x

        expected_hessian = compute_hessian(f, [x, y])
        expected_hvp = tf.matmul(vector, expected_hessian)
        reg_coeff = 1e-5
        hvp = FiniteDifferenceHvp()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(x.assign([x_val]))
        self.sess.run(y.assign([y_val]))
        hvp.update_hvp(f, policy, (a, b), reg_coeff)
        hx = hvp.build_eval((np.array(a_val), np.array(b_val)))
        hvp = hx(vector[0])
        expected_hvp = expected_hvp.eval()
        np.allclose(hvp, expected_hvp)

    def test_pickleable(self):
        policy = HelperPolicy(n_vars=1)
        x = policy.get_params()[0]
        a_val = np.array([5.0])
        a = tf.constant([0.0])
        f = a * (x**2)
        vector = np.array([10.0])
        reg_coeff = 1e-5
        hvp = FiniteDifferenceHvp()

        self.sess.run(tf.global_variables_initializer())
        hvp.update_hvp(f, policy, (a, ), reg_coeff)
        hx = hvp.build_eval(np.array([a_val]))
        before_pickle = hx(vector)

        hvp = pickle.loads(pickle.dumps(hvp))
        hvp.update_hvp(f, policy, (a, ), reg_coeff)
        after_pickle = hx(vector)
        assert np.equal(before_pickle, after_pickle)


def compute_hessian(f, params):
    h = []
    for i in params:
        h_i = []
        for j in params:
            h_ij = tf.gradients(tf.gradients(f, j)[0], i)[0]
            h_ij = [0.] if h_ij is None else h_ij
            h_i.append(h_ij)
        h_i = tf.convert_to_tensor(h_i)
        h.append(h_i)
    h = tf.convert_to_tensor(h)
    h = tf.reshape(h, (len(params), len(params)))
    return h
