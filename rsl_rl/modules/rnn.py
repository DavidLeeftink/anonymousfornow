# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules.cpg import HopfOscillatorRNN
from rsl_rl.modules.ncp import NCPMamba, MinGRU, MinLSTM
from rsl_rl.modules.ncp import ComplexSelective, S4D, LRU
from typing import Union

from rsl_rl.utils import unpad_trajectories

HiddenState = Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor], None]  # Using Union due to Python <3.10
"""Type alias for the hidden state of RNNs (GRU/LSTM).

For GRUs, this is a single tensor while for LSTMs, this is a tuple of two tensors (hidden state and cell state).
"""


_CELL_REGISTRY: dict[str, type] = {
    "gru": nn.GRU,
    "lstm": nn.LSTM,
    "mingru": MinGRU,
    "minlstm": MinLSTM,
    "ncp-mamba": NCPMamba,
    "s4d": S4D,
    "lru": LRU,
}
"""Recurrent cell classes by config name. Every arm of a sweep must appear here."""

_READOUT_CELLS = {"ncp-mamba", "complex-selective", "s4d", "lru"}


class RNN(nn.Module):
    """Recurrent Neural Network.

    This network is used to store the hidden state of the policy.
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        type: str = "lstm",
        **cell_kwargs,
    ) -> None:
        """Initialize a recurrent module with internal hidden-state storage.

        Args:
            input_size: Input dimension.
            hidden_dim: Recurrent width. NOTE this is not the size of the stored
                hidden state for every cell -- see the `state_size` property.
            num_layers: Number of stacked layers.
            type: Cell name, a key of `_CELL_REGISTRY`.
            **cell_kwargs: Cell-specific options, e.g. `dt_env` for the SSM
                cells (which must match the environment control period, or
                every timescale prior is wrong by that ratio), `state_expansion`
                for S4D, `oscillatory_fraction` for ComplexSelective.
        """
        super().__init__()
        key = type.lower()
        if key not in _CELL_REGISTRY:
            raise NotImplementedError(
                f"No correct cell type found: '{type}'. Available: {sorted(_CELL_REGISTRY)}"
            )
        self.type = key
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        rnn_cls = _CELL_REGISTRY[key]
        if rnn_cls in (nn.GRU, nn.LSTM) and cell_kwargs:
            # Silently dropping these would let a sweep arm claim options it
            # never applied, which looks like data.
            raise ValueError(f"'{type}' accepts no cell options, got {sorted(cell_kwargs)}.")
        self.rnn = rnn_cls(
            input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, **cell_kwargs
        )
        self.hidden_state = None
        self._costate_sequence: torch.Tensor | None = None

    @property
    def state_size(self) -> int:
        """Reals per layer per sample carried in the hidden state.

        This is NOT `hidden_dim` in general: LSTM carries h and c, and the
        complex cells store [Re h, Im h]. Anything that allocates, reshapes or
        registers a buffer for the hidden state must use this.
        """
        if self.type == "lstm":
            return 2 * self.hidden_dim
        if hasattr(self.rnn, "state_size"):
            return self.rnn.state_size
        return self.hidden_dim

    @property
    def costate_size(self) -> int:
        """Width of the tensor `costate_state()` returns.

        Differs from `state_size` for LSTM: the stored hidden state is the
        (h, c) tuple, but the captured per-timestep sequence is the output
        h_t = o_t * tanh(c_t), of width hidden_dim. The cell state is not
        exposed by the fused cuDNN kernel.
        """
        if self.type == "lstm":
            return self.hidden_dim
        return self.state_size

    def enable_costate_capture(self) -> None:
        """Opt in to recording the hidden sequence. Fails at construction, not mid-run."""
        if self.type in _READOUT_CELLS and not hasattr(self.rnn, "costate_state"):
            raise NotImplementedError(
                f"'{self.type}' is a readout cell but {type(self.rnn).__name__} "
                "does not implement costate_state()."
            )
        if self.type not in _READOUT_CELLS and self.type not in ("gru", "lstm", "ncp-gru", "mingru", "minlstm"):
            raise NotImplementedError(f"No co-state source defined for cell '{self.type}'.")

    def costate_state(self) -> torch.Tensor:
        """Per-timestep hidden state h for the co-state loss, [T, B, state_size].

        Valid only after a forward pass; in batch (update) mode the sequence is
        unpadded to match the returned output.
        """
        if self._costate_sequence is None:
            raise RuntimeError(
                "costate_state() called before forward(), or after a forward "
                "pass by a cell that does not expose its hidden sequence."
            )
        return self._costate_sequence

    def forward(
        self,
        input: torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Run recurrent inference in rollout mode or batched update mode."""
        self._costate_sequence = None  # never serve a stale sequence
        batch_mode = masks is not None
        if batch_mode:
            # Batch mode needs saved hidden states
            if hidden_state is None:
                raise ValueError("Hidden states not passed to RNN module during policy update")
            out, _ = self.rnn(input, hidden_state)
            self._capture_costate_sequence(out, masks)
            out = unpad_trajectories(out, masks)
        else:
            # Inference/distillation mode uses hidden state of last step
            out, self.hidden_state = self.rnn(input.unsqueeze(0), self.hidden_state)
            self._capture_costate_sequence(out, None)
        return out

    def _capture_costate_sequence(self, out: torch.Tensor, masks: torch.Tensor | None) -> None:
        """Store the hidden sequence the co-state loss should see.

        For readout cells this comes from the cell wrapper; for the rest the
        output already is h. Unpadded in batch mode so it aligns index-for-index
        with the unpadded output the loss is otherwise computed against.
        """
        if self.type in _READOUT_CELLS:
            if not hasattr(self.rnn, "costate_state"):
                # Not fatal here -- only a run that actually enables the
                # co-state loss needs this -- so defer the error to the getter.
                return
            seq = self.rnn.costate_state()
        else:
            seq = out
        self._costate_sequence = unpad_trajectories(seq, masks) if masks is not None else seq

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset hidden states for all or done environments."""
        if dones is None:  # Reset hidden state
            if hidden_state is None:
                self.hidden_state = None
            else:
                self.hidden_state = hidden_state
        elif self.hidden_state is not None:  # Reset hidden state of done environments
            if hidden_state is None:
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    for hidden_state in self.hidden_state:
                        hidden_state[..., dones == 1, :] = 0.0  # type: ignore
                else:
                    self.hidden_state[..., dones == 1, :] = 0.0
            else:
                raise NotImplementedError(
                    "Resetting the hidden state of done environments with a custom hidden state is not implemented"
                )

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach hidden states for all or done environments from the computation graph."""
        if self.hidden_state is not None:
            if dones is None:  # Detach hidden state
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    self.hidden_state = tuple(hidden_state.detach() for hidden_state in self.hidden_state)
                else:
                    self.hidden_state = self.hidden_state.detach()
            else:  # Detach hidden state of done environments
                if isinstance(self.hidden_state, tuple):  # Tuple in case of LSTM
                    for hidden_state in self.hidden_state:
                        hidden_state[..., dones == 1, :] = hidden_state[..., dones == 1, :].detach()
                else:
                    self.hidden_state[..., dones == 1, :] = self.hidden_state[..., dones == 1, :].detach()