"""Differentiable Stochastic Gradient Descent Optimizer.

Useful for algorithms such as MAML that needs the gradient of functions of
post-updated parameters with respect to pre-updated parameters.

"""


class DifferentiableSGD:
    """Differentiable Stochastic Gradient Descent.

    DifferentiableSGD performs the same optimization step as SGD, but instead
    of updating parameters in-place, it saves updated parameters in new
    tensors, so that the gradient of functions of new parameters can flow back
    to the pre-updated parameters.

    Args:
        module (torch.nn.module): A torch module whose parameters needs to be
            optimized.
        lr (float): Learning rate of stochastic gradient descent.

    """

    def __init__(self, module, lr=1e-3):
        self.module = module
        self.lr = lr

    def step(self):
        """Take an optimization step."""
        memo = set()

        def update(module):
            for child in module.children():
                if child not in memo:
                    memo.add(child)
                    update(child)

            params = list(module.named_parameters())
            for name, param in params:
                # Skip descendant modules' parameters.
                if '.' not in name:
                    if param.grad is None:
                        continue

                    # Original SGD uses param.grad.data
                    new_param = param.add(param.grad, alpha=-self.lr)

                    del module._parameters[name]  # pylint: disable=protected-access # noqa: E501
                    setattr(module, name, new_param)
                    module._parameters[name] = new_param  # pylint: disable=protected-access # noqa: E501

        update(self.module)

    def zero_grad(self):
        """Sets gradients of all model parameters to zero."""
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def set_grads_none(self):
        """Sets gradients for all model parameters to None.

        This is an alternative to `zero_grad` which sets
        gradients to zero.
        """
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad = None
