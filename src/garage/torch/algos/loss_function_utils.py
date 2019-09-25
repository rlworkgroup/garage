"""loss function utilities."""
import torch
import torch.nn.functional as F


def compute_advantages(discount, gae_lambda, max_len, baselines, rewards):
    """Calculate advantages."""
    filter = torch.full((1, 1, 1, max_len - 1), discount * gae_lambda)
    filter = torch.cumprod(F.pad(filter, (1, 0), value=1), dim=-1)

    deltas = (rewards + discount * F.pad(baselines, (0, 1))[:, 1:] - baselines)
    deltas = F.pad(deltas, (0, max_len - 1)).unsqueeze(0).unsqueeze(0)

    advantages = F.conv2d(deltas, filter, stride=1).squeeze()
    return advantages
