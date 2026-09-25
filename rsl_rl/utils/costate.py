
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn


class CostateProbe(nn.Module):
    r"""Learned linear decoder of the Pontryagin co-state from the hidden state.

        pred_t = W h_t + b,          W in R^{d_lambda x d_h}
        loss   = mean_j  E_t[ (pred_tj - lambda_tj)^2 / var_j ]

    """

    def __init__(
        self,
        hidden_dim: int,
        target_dim: int,
        momentum: float = 0.01,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.probe = nn.Linear(hidden_dim, target_dim, bias=True)
        nn.init.orthogonal_(self.probe.weight, gain=1.0)
        nn.init.zeros_(self.probe.bias)

        self.momentum = momentum
        self.eps = eps
        self.register_buffer("lam_mean", torch.zeros(target_dim))
        self.register_buffer("lam_var", torch.ones(target_dim))
        self.register_buffer("h_mean", torch.zeros(hidden_dim))
        self.register_buffer("h_var", torch.ones(hidden_dim))
        self.register_buffer("warmed_up", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def _update_stats(self, h: torch.Tensor, lam: torch.Tensor) -> None:
        stats = ((self.h_mean, self.h_var, h), (self.lam_mean, self.lam_var, lam))
        if not bool(self.warmed_up):
            # Seed from the first batch. Starting at (0, 1) makes the first
            # losses meaningless whenever the true scale is far from unit.
            for mean_buf, var_buf, x in stats:
                mean_buf.copy_(x.mean(0))
                var_buf.copy_(x.var(0, unbiased=False))
            self.warmed_up.fill_(True)
        else:
            m = self.momentum
            for mean_buf, var_buf, x in stats:
                mean_buf.mul_(1 - m).add_(x.mean(0), alpha=m)
                var_buf.mul_(1 - m).add_(x.var(0, unbiased=False), alpha=m)

    def forward(self, h: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """h: [N, d_h] carrying grad. lam: [N, d_lambda], treated as constant."""
        if h.shape[0] != lam.shape[0]:
            raise ValueError(f"h has {h.shape[0]} rows, lambda has {lam.shape[0]}.")

        lam = lam.detach()
        if self.training:
            self._update_stats(h.detach(), lam)

        h_std = (h - self.h_mean) * torch.rsqrt(self.h_var + self.eps)
        target = (lam - self.lam_mean) * torch.rsqrt(self.lam_var + self.eps)
        return (self.probe(h_std) - target).pow(2).mean()