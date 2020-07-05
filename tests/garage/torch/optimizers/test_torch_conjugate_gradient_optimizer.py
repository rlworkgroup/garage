"""Tests for garage.torch.optimizers.conjugateGradientOptimizer."""
# yapf: disable
import pickle

import numpy as np
import pytest
import torch

from garage.torch.optimizers.conjugate_gradient_optimizer import (
    ConjugateGradientOptimizer)  # noqa: E501
from garage.torch.optimizers.conjugate_gradient_optimizer import (
    _build_hessian_vector_product)  # noqa: E501
from garage.torch.optimizers.conjugate_gradient_optimizer import (
    _conjugate_gradient)  # noqa: E501

# yapf: enable

# pylint: disable=not-callable  #https://github.com/pytorch/pytorch/issues/24807  # noqa: E501


class TestTorchConjugateGradientOptimizer:
    """Test class for conjugate gradient optimizer."""

    def test_line_search_should_stop(self):
        """Test if line search stops when loss is decreasing, and constraint is satisfied."""  # noqa: E501
        p1 = torch.tensor([0.1])
        p2 = torch.tensor([0.1])
        params = [p1, p2]
        optimizer = ConjugateGradientOptimizer(params, 0.01)
        expected_num_steps = 1

        loss_calls = 0
        first_time = True

        def f_loss():
            nonlocal loss_calls, first_time
            if first_time:
                first_time = False
            else:
                loss_calls += 1

            return -torch.tensor(loss_calls)

        kl_calls = 0

        def f_constrint():
            nonlocal kl_calls
            kl_calls += 1
            return -torch.tensor(kl_calls)

        descent_step = torch.tensor([0.05, 0.05])
        optimizer._backtracking_line_search(params, descent_step, f_loss,
                                            f_constrint)

        assert loss_calls == expected_num_steps
        assert kl_calls == expected_num_steps

    def test_line_search_step_size_should_decrease(self):
        """Line search step size should always decrease."""
        p1 = torch.tensor([0.1])
        p2 = torch.tensor([0.1])
        params = [p1, p2]
        optimizer = ConjugateGradientOptimizer(params, 0.01)
        p1_history = []
        p2_history = []
        loss = 0

        first_time = True

        def f_loss():
            nonlocal loss, first_time
            if first_time:
                first_time = False
            else:
                p1_history.append(p1.clone())
                p2_history.append(p2.clone())
                loss += 1

            return torch.tensor(loss)

        def f_constrint():
            return torch.tensor(0)

        descent_step = torch.tensor([0.05, 0.05])
        optimizer._backtracking_line_search(params, descent_step, f_loss,
                                            f_constrint)

        p1_steps = []
        p2_steps = []
        for i in range(len(p1_history) - 1):
            p1_steps.append(p1_history[i + 1] - p1_history[i])
            p2_steps.append(p2_history[i + 1] - p2_history[i])

        for i in range(len(p1_steps) - 1):
            assert p1_steps[i] > p1_steps[i + 1]
            assert p2_steps[i] > p2_steps[i + 1]


def test_cg():
    """Solve Ax = b using Conjugate gradient method."""
    a = np.linspace(-np.pi, np.pi, 25).reshape((5, 5))
    a = a.T.dot(a)  # make sure a is positive semi-definite

    def hvp(v):
        return torch.tensor(a.dot(v))

    b = torch.tensor(np.linspace(-np.pi, np.pi, 5))
    x = _conjugate_gradient(hvp, b, 5)
    assert np.allclose(a.dot(x), b)


def test_hessian_vector_product():
    """Test Hessian-vector product for a function with one variable."""
    a = torch.tensor([5.0])
    x = torch.tensor([10.0], requires_grad=True)

    def f():
        return a * (x**2)

    expected_hessian = 2 * a
    vector = torch.tensor([10.0])
    expected_hvp = (expected_hessian * vector).detach()
    f_Ax = _build_hessian_vector_product(f, [x])
    computed_hvp = f_Ax(vector).detach()
    assert np.allclose(computed_hvp, expected_hvp)


@pytest.mark.parametrize('a_val, b_val, x_val, y_val, vector', [
    (1.0, 1.0, 1.0, 1.0, [10.0, 20.0]),
    (5.0, 10.0, -2.0, 5.0, [0.0, -1.0]),
    (0.0, 0.0, 1.1, 0.02, [0.0, 0.0]),
    (-2.2, -1.5, -12.3, 34.8, [2.2, 5.3]),
    (-1.5, 0.0, -0.002, 4.93, [0.1, -0.02]),
])
def test_hessian_vector_product_2x2(a_val, b_val, x_val, y_val, vector):
    """Test for a function with two variables."""
    obs = [torch.tensor([a_val]), torch.tensor([b_val])]
    vector = torch.tensor([vector])
    x = torch.tensor(x_val, requires_grad=True)
    y = torch.tensor(y_val, requires_grad=True)

    def f():
        a, b = obs[0], obs[1]
        return a * (x**2) + b * (y**2)

    expected_hessian = compute_hessian(f(), [x, y])
    expected_hvp = torch.mm(vector, expected_hessian).detach()
    f_Ax = _build_hessian_vector_product(f, [x, y])
    hvp = f_Ax(vector[0]).detach()
    assert np.allclose(hvp, expected_hvp, atol=1e-6)


@pytest.mark.parametrize('a_val, b_val, x_val, y_val, vector', [
    (1.0, 1.0, 1.0, 1.0, [10.0, 20.0]),
    (5.0, 10.0, -2.0, 5.0, [0.0, -1.0]),
    (0.0, 0.0, 1.1, 0.02, [0.0, 0.0]),
    (-2.2, -1.5, -12.3, 34.8, [2.2, 5.3]),
    (-1.5, 0.0, -0.002, 4.93, [0.1, -0.02]),
])
def test_hessian_vector_product_2x2_non_diagonal(a_val, b_val, x_val, y_val,
                                                 vector):
    """Test for a function with two variables and non-diagonal Hessian."""
    obs = [torch.tensor([a_val]), torch.tensor([b_val])]
    vector = torch.tensor([vector])
    x = torch.tensor([x_val], requires_grad=True)
    y = torch.tensor([y_val], requires_grad=True)

    def f():
        a, b = obs[0], obs[1]
        kl = a * (x**3) + b * (y**3) + (x**2) * y + (y**2) * x
        return kl

    expected_hessian = compute_hessian(f(), [x, y])
    expected_hvp = torch.mm(vector, expected_hessian).detach()
    f_Ax = _build_hessian_vector_product(f, [x, y])
    hvp = f_Ax(vector[0]).detach()
    assert np.allclose(hvp, expected_hvp)


def compute_hessian(f, params):
    """Compute hessian matrix of given function."""
    h = []
    for i in params:
        h_i = []
        for j in params:
            grad = torch.autograd.grad(f, j, create_graph=True)
            h_ij = torch.autograd.grad(grad,
                                       i,
                                       allow_unused=True,
                                       retain_graph=True)
            h_ij = (torch.tensor(0.), ) if h_ij[0] is None else h_ij
            h_i.append(h_ij[0])
        h_i = torch.stack(h_i)
        h.append(h_i)
    h = torch.stack(h)
    h = h.reshape((len(params), len(params)))
    return h


def test_pickle_round_trip():
    """Test that pickling works as one would normally expect."""
    # pylint: disable=protected-access
    p1 = torch.tensor([0.1])
    p2 = torch.tensor([0.1])
    params = [p1, p2]
    optimizer = ConjugateGradientOptimizer(params, 0.01)
    optimizer_pickled = pickle.dumps(optimizer)
    optimizer2 = pickle.loads(optimizer_pickled)
    assert optimizer._max_constraint_value == optimizer2._max_constraint_value
    assert optimizer._cg_iters == optimizer2._cg_iters
    assert optimizer._max_backtracks == optimizer2._max_backtracks
    assert optimizer._backtrack_ratio == optimizer2._backtrack_ratio
    assert optimizer._hvp_reg_coeff == optimizer2._hvp_reg_coeff
    assert optimizer._accept_violation == optimizer2._accept_violation


class BrokenPicklingConjugateGradientOptimizer(ConjugateGradientOptimizer):
    """Used to check unpickling compat with old implementation."""

    @property
    def state(self):
        """dict: Get the (empty) state."""
        return dict()

    @state.setter
    def state(self, state):
        # Pylint is confused and thinks fset doesn't exist.
        # pylint: disable=no-member
        ConjugateGradientOptimizer.state.fset(self, state)


def test_unpickle_empty_state():
    """Test that pickling works as one would normally expect."""
    # pylint: disable=protected-access
    p1 = torch.tensor([0.1])
    p2 = torch.tensor([0.1])
    params = [p1, p2]
    optimizer = BrokenPicklingConjugateGradientOptimizer(params, 0.02)
    optimizer_pickled = pickle.dumps(optimizer)
    optimizer2 = pickle.loads(optimizer_pickled)
    assert optimizer2._max_constraint_value == 0.01
    # These asserts only pass because they contain the default values.
    assert optimizer._cg_iters == optimizer2._cg_iters
    assert optimizer._max_backtracks == optimizer2._max_backtracks
    assert optimizer._backtrack_ratio == optimizer2._backtrack_ratio
    assert optimizer._hvp_reg_coeff == optimizer2._hvp_reg_coeff
    assert optimizer._accept_violation == optimizer2._accept_violation
