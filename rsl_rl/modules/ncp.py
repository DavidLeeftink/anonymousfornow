from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MinGRUCell(nn.Module):
    """
    Mathematical formulation:
        z_t = sigmoid(Linear(x_t))
        h_tilde = Linear(x_t)
        h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Only 2 components needed: Update gate (z) and Candidate (h_tilde)
        self.weight_ih = nn.Parameter(torch.empty(2 * hidden_size, input_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(2 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        if self.weight_ih is not None:
            nn.init.uniform_(self.weight_ih, -stdv, stdv)
        if self.bias_ih is not None:
            nn.init.uniform_(self.bias_ih, -stdv, stdv)

    def forward(self, input: torch.Tensor, hx: torch.Tensor | None = None) -> torch.Tensor:
        is_unbatched = input.dim() == 1
        if is_unbatched:
            input = input.unsqueeze(0)
            
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        elif is_unbatched:
            hx = hx.unsqueeze(0)

        # 1. External Input Projections ONLY (No recurrent weights)
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        
        # 2. Slice into components
        i_z, i_h = gi.chunk(2, dim=-1)
        
        # 3. Compute Gate and Candidate (No tanh on candidate!)
        z_t = torch.sigmoid(i_z)        
        h_tilde = i_h                   
        
        # 4. Final state integration
        h_next = (1.0 - z_t) * hx + z_t * h_tilde
        
        if is_unbatched:
            h_next = h_next.squeeze(0)
            
        return h_next


class MinGRU(nn.Module):
    """
    Matches the exact input/output signatures of torch.nn.GRU.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1, 
        bias: bool = True, 
        batch_first: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        
        self.cells = nn.ModuleList([
            MinGRUCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            for i in range(num_layers)
        ])

    def forward(self, input: torch.Tensor, hx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            input = input.transpose(0, 1)
            
        seq_len, batch_size, _ = input.shape
        
        if hx is None:
            hx = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
            
        current_hidden = [hx[l] for l in range(self.num_layers)]
        output_sequence = []

        for t in range(seq_len):
            layer_input = input[t]
            
            for l, cell in enumerate(self.cells):
                current_hidden[l] = cell(layer_input, current_hidden[l])
                layer_input = current_hidden[l]
                
                if l < self.num_layers - 1 and self.dropout > 0.0 and self.training:
                    layer_input = F.dropout(layer_input, p=self.dropout, training=self.training)
            
            output_sequence.append(current_hidden[-1])

        output = torch.stack(output_sequence, dim=0)
        final_hidden = torch.stack(current_hidden, dim=0)
        
        if self.batch_first:
            output = output.transpose(0, 1)
            
        return output, final_hidden


class MinLSTMCell(nn.Module):
    """
    Mathematical formulation:
        f_t = sigmoid(Linear(x_t))
        i_t = sigmoid(Linear(x_t))
        h_tilde = Linear(x_t)
        
        f_prime = f_t / (f_t + i_t)
        i_prime = i_t / (f_t + i_t)
        
        h_t = f_prime * h_{t-1} + i_prime * h_tilde
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 3 components needed: Forget gate (f), Input gate (i), and Candidate (h_tilde)
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        if self.weight_ih is not None:
            nn.init.uniform_(self.weight_ih, -stdv, stdv)
        if self.bias_ih is not None:
            nn.init.uniform_(self.bias_ih, -stdv, stdv)
            
            # Initialize forget gate bias to 1.0 for early memory preservation
            with torch.no_grad():
                self.bias_ih[:self.hidden_size].fill_(1.0)

    def forward(self, input: torch.Tensor, hx: torch.Tensor | None = None) -> torch.Tensor:
        is_unbatched = input.dim() == 1
        if is_unbatched:
            input = input.unsqueeze(0)
            
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        elif is_unbatched:
            hx = hx.unsqueeze(0)

        # 1. Linear projections
        gi = F.linear(input, self.weight_ih, self.bias_ih)
        
        # 2. Slice into components
        i_f, i_i, i_h = gi.chunk(3, dim=-1)
        
        # 3. Compute Raw Gates and Candidate
        f_t_raw = torch.sigmoid(i_f)
        i_t_raw = torch.sigmoid(i_i)
        h_tilde = i_h
        
        # 4. Normalize the gates
        gate_sum = f_t_raw + i_t_raw + 1e-6 
        f_prime = f_t_raw / gate_sum
        i_prime = i_t_raw / gate_sum
        
        # 5. Integration
        h_next = f_prime * hx + i_prime * h_tilde
        
        if is_unbatched:
            h_next = h_next.squeeze(0)
            
        return h_next


class MinLSTM(nn.Module):
    """
    Matches the exact input/output signatures of torch.nn.LSTM, but returns 
    a single hidden state instead of an (h, c) tuple.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1, 
        bias: bool = True, 
        batch_first: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        
        self.cells = nn.ModuleList([
            MinLSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            for i in range(num_layers)
        ])

    def forward(self, input: torch.Tensor, hx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            input = input.transpose(0, 1)
            
        seq_len, batch_size, _ = input.shape
        
        if hx is None:
            hx = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
            
        current_hidden = [hx[l] for l in range(self.num_layers)]
        output_sequence = []

        for t in range(seq_len):
            layer_input = input[t]
            
            for l, cell in enumerate(self.cells):
                current_hidden[l] = cell(layer_input, current_hidden[l])
                layer_input = current_hidden[l]
                
                if l < self.num_layers - 1 and self.dropout > 0.0 and self.training:
                    layer_input = F.dropout(layer_input, p=self.dropout, training=self.training)
            
            output_sequence.append(current_hidden[-1])

        output = torch.stack(output_sequence, dim=0)
        final_hidden = torch.stack(current_hidden, dim=0)
        
        if self.batch_first:
            output = output.transpose(0, 1)
            
        return output, final_hidden


class NCPMambaCell(nn.Module):
    """S6 Selective diagonal SSM cell, N=1, affine in the state.
 
        h_t = exp(-delta_t * nu) * h_{t-1} + delta_t * u_t
        y_t = RMSNorm( h_t * C_t + D * u_t )
 
    All of (delta_t, u_t, C_t) are functions of the input only, so the
    recurrence is affine in h with input-varying coefficients.
 
    """
 
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_expansion: int = 1,
        bias: bool = True,
        dt_env: float = 0.02,          # control period in seconds (50 Hz)
        bands: tuple | None = None,    # (tau_lo_s, tau_hi_s, fraction) per band
    ):
        super().__init__()
        if state_expansion != 1:
            raise ValueError(
                "NCPMambaCell is N=1 only. The timescale prior and every "
                "per-mode diagnostic assume one filter per channel."
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_expansion = 1
        self.dt_env = dt_env
        self.bands = bands or (
            (0.04, 0.20, 0.25),   # 
            (0.20, 2.00, 0.50),   # 
            (2.00, 20.0, 0.25),   # 
        )
 
        self.proj_u = nn.Linear(input_size, hidden_size, bias=bias)
        self.proj_delta = nn.Linear(input_size, hidden_size, bias=True)
        self.proj_C = nn.Linear(input_size, hidden_size, bias=bias)
 
        self.A_log = nn.Parameter(torch.zeros(hidden_size))   # nu = exp(A_log)
        self.D = nn.Parameter(torch.ones(hidden_size))
 
        self.norm = RMSNorm(hidden_size)
        self.reset_parameters()
 
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.proj_u.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj_C.weight, a=math.sqrt(5))
        if self.proj_u.bias is not None:
            nn.init.zeros_(self.proj_u.bias)
        if self.proj_C.bias is not None:
            nn.init.zeros_(self.proj_C.bias)
 
        # nu = 1 for every channel, so tau = 1 / delta exactly.
        nn.init.zeros_(self.A_log)
 
        # Banded log-uniform timescale prior.
        n_ch, taus = self.hidden_size, []
        for lo, hi, frac in self.bands:
            k = max(1, int(round(frac * n_ch)))
            taus.append(torch.logspace(
                math.log10(lo / self.dt_env), math.log10(hi / self.dt_env), k
            ))
        taus = torch.cat(taus)
        if taus.numel() < n_ch:   # rounding shortfall
            taus = torch.cat([taus, taus[-1].repeat(n_ch - taus.numel())])
        taus = taus[:n_ch][torch.randperm(n_ch)]   # decorrelate band from index
 
        delta0 = 1.0 / taus
        self.proj_delta.bias.data.copy_(torch.log(torch.expm1(delta0)))
 
        # Shrink the input weights or the bias prior is swamped by noise at init.
        nn.init.kaiming_uniform_(self.proj_delta.weight, a=math.sqrt(5))
        self.proj_delta.weight.data.mul_(0.1)
 
        nn.init.ones_(self.D)
 
    @torch.no_grad()
    def timescales(self) -> torch.Tensor:
        """Nominal time constants in seconds at the bias operating point."""
        delta0 = F.softplus(self.proj_delta.bias)
        return self.dt_env / (delta0 * torch.exp(self.A_log))
 
    @torch.no_grad()
    def spectral_radius(self, input: torch.Tensor) -> torch.Tensor:
        """Per-channel |A_t| for the given input. Bounds error propagation."""
        delta = F.softplus(self.proj_delta(input))
        return torch.exp(-delta * torch.exp(self.A_log))
 
    def forward(
        self, input: torch.Tensor, hx: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_unbatched = input.dim() == 1
        if is_unbatched:
            input = input.unsqueeze(0)
        batch_size = input.size(0)
 
        if hx is None:
            hx = torch.zeros(
                batch_size, self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            if is_unbatched:
                hx = hx.unsqueeze(0)
            hx = hx.view(batch_size, self.hidden_size)   # already flat at N=1
 
        u_t = self.proj_u(input)                          # [B, D]
        delta = F.softplus(self.proj_delta(input))        # [B, D]
        C_t = self.proj_C(input)                          # [B, D]
 
        nu = torch.exp(self.A_log)                        # [D], A = -nu < 0
        bar_A = torch.exp(-delta * nu)                    # [B, D], in (0, 1)
 
        h_next = bar_A * hx + delta * u_t                 # affine in hx
        y_t = self.norm(h_next * C_t + self.D * u_t)
 
        if is_unbatched:
            h_next = h_next.squeeze(0)
            y_t = y_t.squeeze(0)
        return h_next, y_t

    @property
    def state_size(self) -> int:
        return self.hidden_size          # N=1, real state

 
 
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
 

class S4DCell(nn.Module):
    """Diagonal S4 (S4D), SISO, LTI, per-channel state of size N.
 
        A_n     = -1/2 + i*pi*n                    (S4D-Lin initialization)
        A_bar   = exp(delta * A)                   ZOH, delta learned per channel
        B_bar   = (A_bar - 1) / A                  (S4D-Lin fixes B = 1)
        h_t     = A_bar * h_{t-1} + B_bar * x_t
        y_t     = 2 * Re( sum_n C_n h_t,n ) + D * x_t
 
    Affine in h, so inside the CP class. 
    """
 
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_expansion: int = 8,           # N, modes per channel
        bias: bool = True,
        dt_env: float = 0.02,
        dt_range: tuple = (0.001, 0.1),
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_expansion = state_expansion
        self.dt_env = dt_env
        self.dt_range = dt_range
 
        H, N = hidden_size, state_expansion
 
        # SISO: the input is projected to H channels once, then each channel is its own scalar sequence.
        self.proj_in = nn.Linear(input_size, hidden_size, bias=bias)
 
        # A = -exp(A_log_re) + i*A_im, per (H, N). exp keeps Re(A) < 0, so
        # |A_bar| = exp(delta*Re(A)) < 1 for any parameter value.
        self.A_log_re = nn.Parameter(torch.empty(H, N))
        self.A_im = nn.Parameter(torch.empty(H, N))
        # Complex C, per (H, N).
        self.C_re = nn.Parameter(torch.empty(H, N))
        self.C_im = nn.Parameter(torch.empty(H, N))
        # Per-channel step size, log-parameterized.
        self.log_dt = nn.Parameter(torch.empty(H))
        self.D = nn.Parameter(torch.ones(H))
 
        self.norm = RMSNorm(hidden_size)
        self.reset_parameters()
 
    def reset_parameters(self) -> None:
        H, N = self.hidden_size, self.state_expansion
 
        nn.init.kaiming_uniform_(self.proj_in.weight, a=math.sqrt(5))
        if self.proj_in.bias is not None:
            nn.init.zeros_(self.proj_in.bias)
 
        # S4D-Lin: A_n = -1/2 + i*pi*n, the same spectrum for every channel.
        n = torch.arange(N, dtype=torch.float32)
        self.A_log_re.data.copy_(torch.full((H, N), math.log(0.5)))
        self.A_im.data.copy_((math.pi * n).unsqueeze(0).expand(H, N).contiguous())
 
        # C ~ N(0, 1/2) per real and imaginary part, as in the reference impl.
        self.C_re.data.normal_(0.0, 0.5 ** 0.5)
        self.C_im.data.normal_(0.0, 0.5 ** 0.5)
 
        lo, hi = self.dt_range
        log_dt = torch.rand(H) * (math.log(hi) - math.log(lo)) + math.log(lo)
        self.log_dt.data.copy_(log_dt)
 
        nn.init.ones_(self.D)
  
    @property
    def state_size(self) -> int:
        """Reals carried in the hidden state: [Re h, Im h], each H*N."""
        return 2 * self.hidden_size * self.state_expansion
 
    @torch.no_grad()
    def timescales(self) -> torch.Tensor:
        """Time constants in seconds, [H, N]. One per mode, not per channel."""
        dt = torch.exp(self.log_dt).unsqueeze(-1)          # [H, 1]
        return self.dt_env / (dt * torch.exp(self.A_log_re))
 
    @torch.no_grad()
    def frequencies(self) -> torch.Tensor:
        """Mode frequencies in Hz, [H, N]."""
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        return (dt * self.A_im).abs() / (2.0 * math.pi * self.dt_env)
 
    @torch.no_grad()
    def spectral_radius(self, input: torch.Tensor | None = None) -> torch.Tensor:
        """Per-mode |A_bar|, [H, N]. LTI, so the input is ignored."""
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        return torch.exp(-dt * torch.exp(self.A_log_re))
  
    def forward(
        self, input: torch.Tensor, hx: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One step. hx is [B, 2*H*N], the concatenation [Re h, Im h]."""
        is_unbatched = input.dim() == 1
        if is_unbatched:
            input = input.unsqueeze(0)
        B = input.size(0)
        H, N = self.hidden_size, self.state_expansion
 
        if hx is None:
            hx = torch.zeros(B, 2 * H * N, dtype=input.dtype, device=input.device)
        else:
            if is_unbatched:
                hx = hx.unsqueeze(0)
            hx = hx.reshape(B, 2 * H * N)
        h_re = hx[:, : H * N].reshape(B, H, N)
        h_im = hx[:, H * N :].reshape(B, H, N)
 
        x = self.proj_in(input)                             # [B, H]
 
        dt = torch.exp(self.log_dt).unsqueeze(-1)           # [H, 1]
        A_re = -torch.exp(self.A_log_re)                    # [H, N], < 0
        A_im = self.A_im                                    # [H, N]
 
        # A_bar = exp(dt * A)
        mag = torch.exp(dt * A_re)                          # [H, N], in (0, 1)
        ang = dt * A_im                                     # [H, N]
        Abar_re = mag * torch.cos(ang)
        Abar_im = mag * torch.sin(ang)
 
        # B_bar = (A_bar - 1) / A, with B = 1. Complex divide.
        num_re, num_im = Abar_re - 1.0, Abar_im
        den = A_re * A_re + A_im * A_im
        Bbar_re = (num_re * A_re + num_im * A_im) / den
        Bbar_im = (num_im * A_re - num_re * A_im) / den
 
        # h <- A_bar * h + B_bar * x. Affine in h.
        xb = x.unsqueeze(-1)                                # [B, H, 1]
        h_next_re = Abar_re * h_re - Abar_im * h_im + Bbar_re * xb
        h_next_im = Abar_re * h_im + Abar_im * h_re + Bbar_im * xb
 
        # y = 2 Re(sum_n C_n h_n) + D x. 
        y = 2.0 * (self.C_re * h_next_re - self.C_im * h_next_im).sum(dim=-1)
        y_t = self.norm(y + self.D * x)
 
        h_next = torch.cat(
            [h_next_re.reshape(B, H * N), h_next_im.reshape(B, H * N)], dim=-1
        )
        if is_unbatched:
            h_next = h_next.squeeze(0)
            y_t = y_t.squeeze(0)
        return h_next, y_t


class LRUCell(nn.Module):
    """Linear Recurrent Unit (Orvieto et al., 2023), as published.
 
        A       = exp(-exp(nu_log) + i*exp(theta_log))     diagonal, complex
        gamma   = sqrt(1 - |A|^2)                          input normalization
        h_t     = A * h_{t-1} + gamma * (B x_t)            h complex, size N
        y_t     = Re(C h_t) + D x_t
 
        LTI
    """
 
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_expansion: int = 1,      # ignored; state size is set by state_dim
        bias: bool = True,
        state_dim: int | None = None,  # N. Defaults to hidden_size.
        r_min: float = 0.9,
        r_max: float = 0.999,
        max_phase: float = 6.283185307179586,   # 2*pi, the published default
        dt_env: float = 0.02,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_dim = state_dim if state_dim is not None else hidden_size
        self.dt_env = dt_env
        self.r_min, self.r_max, self.max_phase = r_min, r_max, max_phase
 
        N, H = self.state_dim, hidden_size
 
        self.nu_log = nn.Parameter(torch.empty(N))
        self.theta_log = nn.Parameter(torch.empty(N))
        self.gamma_log = nn.Parameter(torch.empty(N))
 
        self.B_re = nn.Parameter(torch.empty(N, input_size))
        self.B_im = nn.Parameter(torch.empty(N, input_size))
        self.C_re = nn.Parameter(torch.empty(H, N))
        self.C_im = nn.Parameter(torch.empty(H, N))
        self.D = nn.Parameter(torch.ones(H))
        self.skip_proj = None if input_size == H else nn.Linear(input_size, H, bias=False)
 
        self.norm = RMSNorm(hidden_size)
        self.reset_parameters()
 
    @property
    def state_size(self) -> int:
        """Reals carried in the hidden state: [Re h, Im h]."""
        return 2 * self.state_dim
 
    def reset_parameters(self) -> None:
        N, Hin, H = self.state_dim, self.input_size, self.hidden_size
        r_min, r_max = self.r_min, self.r_max
 
        # Magnitudes uniform on the ring [r_min, r_max].
        u1 = torch.rand(N)
        self.nu_log.data.copy_(
            torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        )
        # Phases uniform on [0, max_phase].
        u2 = torch.rand(N)
        self.theta_log.data.copy_(torch.log(self.max_phase * u2 + 1e-8))
 
        # gamma = sqrt(1 - |A|^2), stored in log space.
        with torch.no_grad():
            mag = torch.exp(-torch.exp(self.nu_log))
            self.gamma_log.data.copy_(0.5 * torch.log(1.0 - mag ** 2 + 1e-8))
 
        self.B_re.data.normal_(0.0, (1.0 / (2 * Hin)) ** 0.5)
        self.B_im.data.normal_(0.0, (1.0 / (2 * Hin)) ** 0.5)
        self.C_re.data.normal_(0.0, (1.0 / N) ** 0.5)
        self.C_im.data.normal_(0.0, (1.0 / N) ** 0.5)
        nn.init.ones_(self.D)
 
    @torch.no_grad()
    def timescales(self) -> torch.Tensor:
        """Time constants in seconds. |A| = exp(-exp(nu_log)) per step."""
        return self.dt_env / torch.exp(self.nu_log)
 
    @torch.no_grad()
    def frequencies(self) -> torch.Tensor:
        """Mode frequencies in Hz. Phase advance per step is exp(theta_log)."""
        return torch.exp(self.theta_log) / (2.0 * math.pi * self.dt_env)
 
    @torch.no_grad()
    def spectral_radius(self, input: torch.Tensor | None = None) -> torch.Tensor:
        """Per-mode |A|. LTI, so the input is ignored."""
        return torch.exp(-torch.exp(self.nu_log))
 
    def forward(
        self, input: torch.Tensor, hx: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_unbatched = input.dim() == 1
        if is_unbatched:
            input = input.unsqueeze(0)
        B = input.size(0)
        N = self.state_dim
 
        if hx is None:
            hx = torch.zeros(B, 2 * N, dtype=input.dtype, device=input.device)
        else:
            if is_unbatched:
                hx = hx.unsqueeze(0)
            hx = hx.reshape(B, 2 * N)
        h_re, h_im = hx[:, :N], hx[:, N:]
 
        mag = torch.exp(-torch.exp(self.nu_log))            # [N], in (0, 1)
        ang = torch.exp(self.theta_log)                     # [N]
        A_re, A_im = mag * torch.cos(ang), mag * torch.sin(ang)
        gamma = torch.exp(self.gamma_log)                   # [N]
 
        # gamma * (B x), complex.
        Bx_re = F.linear(input, self.B_re) * gamma
        Bx_im = F.linear(input, self.B_im) * gamma
 
        h_next_re = A_re * h_re - A_im * h_im + Bx_re
        h_next_im = A_re * h_im + A_im * h_re + Bx_im
 
        # Re(C h) = C_re h_re - C_im h_im
        y = F.linear(h_next_re, self.C_re) - F.linear(h_next_im, self.C_im)
        skip = self.D * input if self.skip_proj is None else self.skip_proj(input)
        y_t = self.norm(y + skip)
 
        h_next = torch.cat([h_next_re, h_next_im], dim=-1)
        if is_unbatched:
            h_next = h_next.squeeze(0)
            y_t = y_t.squeeze(0)
        return h_next, y_t
 
 
# ---------------------------------------------------------------------------
# Recurrent wrappers
# ---------------------------------------------------------------------------
 
 
class SSMRecurrent(nn.Module):
    """Multi-layer wrapper matching the torch.nn.GRU input/output signature. 
    """
 
    def __init__(
        self,
        cell_cls,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        costate_layer: int = -1,
        **cell_kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.costate_layer = costate_layer
 
        self.cells = nn.ModuleList([
            cell_cls(input_size if i == 0 else hidden_size, hidden_size,
                     bias=bias, **cell_kwargs)
            for i in range(num_layers)
        ])
        self.hidden_sequence: torch.Tensor | None = None
 
    @property
    def state_size(self) -> int:
        return self.cells[0].state_size
 
    def costate_state(self) -> torch.Tensor:
        """The tensor the co-state loss should be applied to, [T, B, state_size].
 
        Raises if forward() has not run, rather than silently returning stale
        activations from a previous rollout.
        """
        if self.hidden_sequence is None:
            raise RuntimeError(
                "costate_state() called before forward(). The hidden sequence "
                "is populated per rollout and is not persisted."
            )
        return self.hidden_sequence
 
    def forward(
        self, input: torch.Tensor, hx: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            input = input.transpose(0, 1)
 
        seq_len, batch_size, _ = input.shape
        S = self.state_size
 
        if hx is None:
            hx = torch.zeros(
                self.num_layers, batch_size, S,
                dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.reshape(self.num_layers, batch_size, S)
 
        current_hidden = [hx[l] for l in range(self.num_layers)]
        output_sequence, hidden_sequence = [], []
        tracked = self.costate_layer % self.num_layers
 
        for t in range(seq_len):
            layer_input = input[t]
            for l, cell in enumerate(self.cells):
                h_next, y_t = cell(layer_input, current_hidden[l])
                current_hidden[l] = h_next
                layer_input = y_t
                if l < self.num_layers - 1 and self.dropout > 0.0 and self.training:
                    layer_input = F.dropout(layer_input, p=self.dropout,
                                            training=self.training)
            output_sequence.append(layer_input)
            hidden_sequence.append(current_hidden[tracked])
 
        output = torch.stack(output_sequence, dim=0)
        self.hidden_sequence = torch.stack(hidden_sequence, dim=0)
        final_hidden = torch.stack(current_hidden, dim=0)
 
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, final_hidden
 
 
def _make(cell_cls):

    class _Wrapped(SSMRecurrent):
        def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                     batch_first=False, dropout=0.0, **kw):
            super().__init__(cell_cls, input_size, hidden_size, num_layers,
                             bias, batch_first, dropout, **kw)
 
    _Wrapped.__name__ = cell_cls.__name__.replace("Cell", "")
    _Wrapped.__qualname__ = _Wrapped.__name__
    return _Wrapped
 
 
ComplexSelective = _make(ComplexSelectiveCell)
NCPMamba = _make(NCPMambaCell)
S4D = _make(S4DCell)
LRU = _make(LRUCell)