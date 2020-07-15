# Behavioral Cloning

[[Paper](https://arxiv.org/abs/1605.08478)] [[Implementation](https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/algos/bc.py)]

Garage's bahavioral cloning algorithm is based on model-free imitation learning with policy optimization. The algorithm uses a stochastic policy or deterministic policy and learns based on sample trajectories from the expert. It operates on SGD and is garanteed to converge to local minimum.

Tuned examples: [bc_point](https://github.com/rlworkgroup/garage/blob/master/examples/torch/bc_point.py), [bc_point_deterministic_policy](https://github.com/rlworkgroup/garage/blob/master/examples/torch/bc_point_deterministic_policy.py)

## Default Parameters

```Python
policy_optimizer = torch.optim.Adam
policy_lr = _Default(1e-3)
loss = 'log_prob' # must be ‘log_prob’ or ‘mse’. If set to ‘log_prob’ (the default), learner must be a garage.torch.StochasticPolicy.
minibatches_per_epoch = 16
```
