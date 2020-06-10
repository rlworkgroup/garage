"""Tests for DifferentialSGD optimizer."""
import torch

from garage.torch import update_module_params
from garage.torch.optimizers import DifferentiableSGD


def test_differentiable_sgd():
    """Test second order derivative after taking optimization step."""
    policy = torch.nn.Linear(10, 10, bias=False)
    lr = 0.01
    diff_sgd = DifferentiableSGD(policy, lr=lr)

    named_theta = dict(policy.named_parameters())
    theta = list(named_theta.values())[0]
    meta_loss = torch.sum(theta**2)
    meta_loss.backward(create_graph=True)

    diff_sgd.step()

    theta_prime = list(policy.parameters())[0]
    loss = torch.sum(theta_prime**2)
    update_module_params(policy, named_theta)
    diff_sgd.zero_grad()
    loss.backward()

    result = theta.grad

    dtheta_prime = 1 - 2 * lr  # dtheta_prime/dtheta
    dloss = 2 * theta_prime  # dloss/dtheta_prime
    expected_result = dloss * dtheta_prime  # dloss/dtheta

    assert torch.allclose(result, expected_result)
