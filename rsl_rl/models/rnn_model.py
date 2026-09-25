# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import RNN, HiddenState
from rsl_rl.modules.readout import build_readout
from rsl_rl.utils import unpad_trajectories


class _HeadAdapter(nn.Module):
    """Presents a Readout as a latent -> output module, pulling obs from the model."""

    def __init__(self, head, model, squash: bool = False):
        super().__init__()
        self.head = head
        self.squash = squash
        self._model = [model]          # list keeps it out of the module tree

    def forward(self, latent):
        obs = self._model[0]._obs_cache if self.head.needs_obs else None
        out = self.head(latent, obs)
        return torch.tanh(out) if self.squash else out

    def init_weights(self, scales):
        self.head.init_weights(scales)


class RNNModel(MLPModel):
    """RNN-based neural model.

    This model uses a recurrent neural network (RNN) to process 1D observation groups before passing the resulting
    latent to an MLP. Available RNN types are "lstm" and "gru". Observations can be normalized before being passed to
    the RNN. The output of the model can be either deterministic or stochastic, in which case a distribution module is
    used to sample the outputs.
    """

    is_recurrent: bool = True
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        squash_output: bool = False,
        encoder: nn.Module | None = None,
        readout: str = "mlp",
    ) -> None:
        """Initialize the RNN-based model.

        Args:
            readout: Head mapping h -> output. "linear" is u = W h + b,
                "mlp" the standard actor head, "costate" the G(y)h form of
                Def. 3.2. `hidden_dims` is used only by "mlp".
        """
        self.latent_dim = rnn_hidden_dim

        # The parent builds an MLP head we discard; hidden_dims must be a
        # non-empty tuple for it to construct, and is only meaningful for the
        # "mlp" readout.
        parent_dims = tuple(hidden_dims) if len(hidden_dims) else (256, 256, 256)

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            parent_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        # --- encoder and recurrence (self.obs_dim exists after super().__init__)
        if encoder is not None:
            enc_in, enc_out = encoder[0].in_features, encoder[0].out_features
            if enc_in != self.obs_dim or enc_out != rnn_hidden_dim:
                raise ValueError(
                    f"Shared encoder is ({enc_in}->{enc_out}) but this model "
                    f"needs ({self.obs_dim}->{rnn_hidden_dim})."
                )
            self.encoder = encoder
        else:
            self.encoder = nn.Sequential(
                nn.Linear(self.obs_dim, rnn_hidden_dim),
                nn.Tanh(),
            )

        self.rnn = RNN(rnn_hidden_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

        # --- readout head, replacing the parent's MLP
        head_out = (
            self.distribution.input_dim if self.distribution is not None else output_dim
        )
        if squash_output and head_out != output_dim:
            raise ValueError(
                "squash_output assumes the head emits only the mean, but this "
                f"distribution wants {head_out} outputs for {output_dim} actions."
            )

        self.readout_type = readout
        self.squash_output = squash_output
        self._obs_cache: torch.Tensor | None = None

        head = build_readout(
            readout,
            latent_dim=self.latent_dim,
            obs_dim=self.obs_dim,
            out_dim=head_out,
            hidden_dims=parent_dims,
            activation=activation,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(head)

        self.mlp = _HeadAdapter(head, self, squash=squash_output)

    def get_encoded_obs(self, obs: TensorDict) -> torch.Tensor:
        """Expose the linear encoder output for HJB co-state regularization."""
        # Extract and concatenate observation groups and normalize
        normalized_obs = super().get_latent(obs) 
        return self.encoder(normalized_obs)
        
    def get_obs_vector(self, obs: TensorDict) -> torch.Tensor:
        """Normalized, concatenated observation groups, pre-encoder."""
        return super().get_latent(obs)

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Build the model latent by passing normalized observation groups through the RNN.
        The pre-encoder observation is cached for readouts that need it
        (CostateReadout), aligned with the latent the head will see.
        """
        obs_vec = self.get_obs_vector(obs)
        latent = self.rnn(self.encoder(obs_vec), masks, hidden_state).squeeze(0)
        self._obs_cache = (
            unpad_trajectories(obs_vec, masks) if masks is not None else obs_vec
        )
        return latent

    def forward_from_latent(self, latent: torch.Tensor, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
        """Forward pass starting directly from the encoded latent representation."""
        # Pass through the RNN
        rnn_out = self.rnn(latent, masks, hidden_state).squeeze(0)
        # Pass through the MLP (inherited from MLPModel)
        mlp_out = self.mlp(rnn_out)
        
        # Return deterministic value output
        if self.distribution is not None:
            return self.distribution.deterministic_output(mlp_out)
        return mlp_out

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the recurrent hidden state of the RNN."""
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state of the RNN."""
        return self.rnn.hidden_state  # type: ignore

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state for truncated backpropagation."""
        self.rnn.detach_hidden_state(dones)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        if isinstance(self.rnn.rnn, nn.LSTM):
            return _TorchLSTMModel(self)
        elif isinstance(self.rnn.rnn, nn.GRU):
            return _TorchGRUModel(self)
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn.rnn)}")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxRNNModel(self, verbose)

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.latent_dim


class _TorchGRUModel(nn.Module):
    """Exportable GRU model for JIT."""

    def __init__(self, model: RNNModel) -> None:
        """Create a TorchScript-friendly copy of a GRU-based RNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.encoder = copy.deepcopy(model.encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one GRU inference step and update hidden states."""
        x = self.obs_normalizer(x)
        x = self.encoder(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset exported GRU hidden states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore


class _TorchLSTMModel(nn.Module):
    """Exportable LSTM model for JIT."""

    def __init__(self, model: RNNModel) -> None:
        """Create a TorchScript-friendly copy of an LSTM-based RNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.encoder = copy.deepcopy(model.encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one LSTM inference step and update hidden and cell states."""
        x = self.obs_normalizer(x)
        x = self.encoder(x)

        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # type: ignore
        self.cell_state[:] = c  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset exported LSTM hidden and cell states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore
        self.cell_state[:] = 0.0  # type: ignore


class _OnnxRNNModel(nn.Module):
    """Exportable RNN model for ONNX."""

    is_recurrent: bool = True

    def __init__(self, model: RNNModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an RNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.encoder = copy.deepcopy(model.encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        # Detect RNN type
        if isinstance(self.rnn, nn.LSTM):
            self.rnn_type = "lstm"
        elif isinstance(self.rnn, nn.GRU):
            self.rnn_type = "gru"
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn)}")

        self.input_size = model.obs_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run deterministic inference for ONNX export."""
        x = self.obs_normalizer(obs)
        x = self.encoder(x)

        if self.rnn_type == "lstm":
            x, (h, c) = self.rnn(x.unsqueeze(0), (h_in, c_in))
            x = x.squeeze(0)
            out = self.mlp(x)
            out = self.deterministic_output(out)
            return out, h, c
        else:
            x, h = self.rnn(x.unsqueeze(0), h_in)
            x = x.squeeze(0)
            out = self.mlp(x)
            out = self.deterministic_output(out)
            return out, h, None

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        obs = torch.zeros(1, self.input_size)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
            return (obs, h_in, c_in)
        return (obs, h_in)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        if self.rnn_type == "lstm":
            return ["obs", "h_in", "c_in"]
        return ["obs", "h_in"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        if self.rnn_type == "lstm":
            return ["actions", "h_out", "c_out"]
        return ["actions", "h_out"]