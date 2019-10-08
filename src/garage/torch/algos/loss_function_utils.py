"""Differentiable utils typically used when computing loss functions."""
import torch
import torch.nn.functional as F


def compute_advantages(discount, gae_lambda, max_path_length, baselines,
                       rewards):
    """Calculate advantages.

    Advantages are a discounted cumulative sum.

    Calculate advantages using a baseline (value function) according to
    Generalized Advantage Estimation (GAE)

    The discounted cumulative sum can be computed using conv2d with filter.
    filter:
        [1, (discount * gae_lambda), (discount * gae_lambda) ^ 2, ...]
        where the length is same with max_path_length.

    baselines and rewards are also has same shape.
        baselines:
        [ [b_11, b_12, b_13, ... b_1n],
          [b_21, b_22, b_23, ... b_2n],
          ...
          [b_m1, b_m2, b_m3, ... b_mn] ]
        rewards:
        [ [r_11, r_12, r_13, ... r_1n],
          [r_21, r_22, r_23, ... r_2n],
          ...
          [r_m1, r_m2, r_m3, ... r_mn] ]

    Args:
        discount (float): RL discount factor (i.e. gamma).
        gae_lambda (float): Lambda, as used for Generalized Advantage
            Estimation (GAE).
        max_path_length (int): Maximum length of a single rollout.
        baselines (torch.Tensor): A 2D vector of value function estimates with
            shape (N, T), where N is the batch dimension (number of episodes)
            and T is the maximum path length experienced by the agent. If an
            episode terminates in fewer than T time steps, the remaining
            elements in that episode should be set to 0.
        rewards (torch.Tensor): A 2D vector of per-step rewards with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum path length experienced by the agent. If an episode
            terminates in fewer than T time steps, the remaining elements in
            that episode should be set to 0.

    Returns:
        torch.Tensor: A 2D vector of calculated advantage values with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum path length experienced by the agent. If an episode
            terminates in fewer than T time steps, the remaining values in that
            episode should be set to 0.

    """
    adv_filter = torch.full((1, 1, 1, max_path_length - 1),
                            discount * gae_lambda)
    adv_filter = torch.cumprod(F.pad(adv_filter, (1, 0), value=1), dim=-1)

    deltas = (rewards + discount * F.pad(baselines, (0, 1))[:, 1:] - baselines)
    deltas = F.pad(deltas, (0, max_path_length - 1)).unsqueeze(0).unsqueeze(0)

    advantages = F.conv2d(deltas, adv_filter, stride=1).squeeze()
    return advantages
