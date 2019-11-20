"""Tests for DifferentialSGD optimizer."""
import torch

from garage.torch.optimizers import DiffSGD


def test_differentiable_sgd():
    """Test second order derivative after taking optimization step."""
    policy = torch.nn.Linear(10, 10, bias=False)
    lr = 0.01
    diff_sgd = DiffSGD(policy, lr=lr)

    theta = list(policy.parameters())
    
    losses = []
    for _ in range(5):
        _theta = list(policy.parameters())[0]

        meta_loss = torch.sum(_theta**2)
        meta_loss.backward(create_graph=True)

        diff_sgd.step()

        theta_prime = list(policy.parameters())[0]
        assert _theta.ne(theta_prime).all()

        loss = torch.sum(theta_prime**2)
        losses.append(loss)
        _update_module_param(policy, theta)

    loss = torch.stack(losses).mean()

    loss.backward()
    result = torch.stack([p.grad for p in theta])

    assert theta_prime.grad is not None

    dtheta_prime = 1 - 2 * lr  # dtheta_prime/dtheta
    dloss = 2 * theta_prime  # dloss/dtheta_prime
    expected_result = dloss * dtheta_prime  # dloss/dtheta
    
    assert torch.allclose(result, expected_result)

def _update_module_param(module, new_params):
    params = list(module.named_parameters())
    #assert len(params) == len(new_params)
    
    for (name, _), new_param in zip(params, new_params):
        if new_param.grad is not None:
            new_param.grad.detach_()
            new_param.grad.zero_()
        module._parameters[name] = new_param  # pylint: disable=protected-access # noqa: E501
