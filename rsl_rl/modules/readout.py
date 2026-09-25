from __future__ import annotations

import math
import torch
import torch.nn as nn

from rsl_rl.modules import MLP


class Readout(nn.Module):
    """Maps the recurrent state to the head output.

    All readouts share forward(h, obs); only the co-state readout uses obs.
    """

    needs_obs: bool = False

    def forward(self, h: torch.Tensor, obs: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class LinearReadout(Readout):
    """u = W h + b. The CP readout when G_theta is constant along the trajectory."""

    def __init__(self, latent_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, out_dim)

    def forward(self, h, obs=None):
        return self.fc(h)

    def init_weights(self, scales):
        nn.init.orthogonal_(self.fc.weight, gain=scales if isinstance(scales, float) else scales[-1])
        nn.init.zeros_(self.fc.bias)


class MLPReadout(Readout):
    """Standard actor head. Nonlinear in h, so outside the CP class."""

    def __init__(self, latent_dim, out_dim, hidden_dims=(256, 256, 256), activation="elu"):
        super().__init__()
        self.net = MLP(latent_dim, out_dim, hidden_dims, activation)

    def forward(self, h, obs=None):
        return self.net(h)

    def init_weights(self, scales):
        self.net.init_weights(scales)


class CostateReadout(Readout):
    """u = W_out (C(y) * h), i.e. G_theta(y) = W_out diag(C(y)).
        Selective readout layer.
    """

    needs_obs = True

    def __init__(self, latent_dim: int, obs_dim: int, out_dim: int):
        super().__init__()
        self.proj_C = nn.Linear(obs_dim, latent_dim, bias=True)
        self.W_out = nn.Linear(latent_dim, out_dim, bias=True)
        nn.init.kaiming_uniform_(self.proj_C.weight, a=math.sqrt(5))
        self.proj_C.weight.data.mul_(0.1)
        nn.init.ones_(self.proj_C.bias)

    def forward(self, h, obs):
        if obs is None:
            raise RuntimeError("CostateReadout requires the observation vector.")
        return self.W_out(self.proj_C(obs) * h)

    def init_weights(self, scales):
        gain = scales if isinstance(scales, float) else scales[-1]
        nn.init.orthogonal_(self.W_out.weight, gain=gain)
        nn.init.zeros_(self.W_out.bias)


def build_readout(kind: str, latent_dim: int, obs_dim: int, out_dim: int,
                  hidden_dims=(256, 256, 256), activation="elu") -> Readout:
    kind = kind.lower()
    if kind == "linear":
        return LinearReadout(latent_dim, out_dim)
    if kind == "mlp":
        return MLPReadout(latent_dim, out_dim, hidden_dims, activation)
    if kind == "costate":
        return CostateReadout(latent_dim, obs_dim, out_dim)
    raise ValueError(f"Unknown readout '{kind}'. Use linear, mlp or costate.")
