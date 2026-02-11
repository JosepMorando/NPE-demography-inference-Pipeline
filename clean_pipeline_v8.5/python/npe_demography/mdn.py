from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


class MDN(nn.Module):
    """Mixture Density Network for p(theta|x) as diagonal Gaussian mixture.

    Outputs K mixture weights, means, and log-stds.
    """

    def __init__(self, x_dim: int, theta_dim: int, hidden: List[int], n_components: int):
        super().__init__()
        layers: List[nn.Module] = []
        d = x_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        self.feature = nn.Sequential(*layers)
        self.pi = nn.Linear(d, n_components)
        self.mu = nn.Linear(d, n_components * theta_dim)
        self.log_sigma = nn.Linear(d, n_components * theta_dim)
        self.theta_dim = theta_dim
        self.k = n_components

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.feature(x)
        logit_pi = self.pi(h)
        pi = torch.softmax(logit_pi, dim=-1)
        mu = self.mu(h).view(-1, self.k, self.theta_dim)
        log_sigma = self.log_sigma(h).view(-1, self.k, self.theta_dim)
        log_sigma = torch.clamp(log_sigma, min=-12.0, max=6.0)
        return pi, mu, log_sigma


def mdn_nll(pi: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood for a diagonal Gaussian mixture."""
    # theta: (B, D)
    theta = theta.unsqueeze(1)  # (B,1,D)
    inv_sigma = torch.exp(-log_sigma)
    z = (theta - mu) * inv_sigma
    # log N(theta|mu,sigma)
    log_norm = -0.5 * torch.sum(z * z, dim=-1) - torch.sum(log_sigma, dim=-1) - 0.5 * mu.shape[-1] * math.log(2 * math.pi)
    # log sum_k pi_k * exp(log_norm_k)
    log_pi = torch.log(pi + 1e-12)
    log_mix = torch.logsumexp(log_pi + log_norm, dim=1)
    return -torch.mean(log_mix)


@torch.no_grad()
def mdn_sample(pi: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, n: int) -> torch.Tensor:
    """Sample n thetas from a batch of MDN outputs. Expects batch size 1."""
    if pi.shape[0] != 1:
        raise ValueError("mdn_sample expects batch size 1")
    pi1 = pi[0]
    mu1 = mu[0]
    sigma1 = torch.exp(log_sigma[0])
    k = pi1.shape[0]
    comp = torch.multinomial(pi1, num_samples=n, replacement=True)
    eps = torch.randn(n, mu1.shape[1], device=mu.device)
    out = mu1[comp] + sigma1[comp] * eps
    return out
