"""Differentiable Stochastic Gradient Descent Optimizer.

Useful for algorithms such as MAML where the function of updated policy
parameters needs to be differentiable with respect to the policy parameters
before optimization update.

"""


class DiffSGD:
    """Differentiable Stochastic Gradient Descent.

    DiffSGD performs the same optimization step as SGD, but instead of updating
    parameters in-place, it creates new parameters such that the gradients
    computed on the function of new parameters can flow back to the original
    parameters.

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

                    new_param = param.add(-self.lr, param.grad)

                    del module._parameters[name]  # pylint: disable=protected-access # noqa: E501
                    setattr(module, name, new_param)
                    module._parameters[name] = new_param  # pylint: disable=protected-access # noqa: E501

        update(self.module)

    def zero_grad(self):
        """Do not need to zero out the gradient.

        Since the optimization update does not happen in-place, new parameters
        created by this optimizer does not have any gradients attached.

        """

