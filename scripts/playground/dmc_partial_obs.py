"""Partial observability for MuJoCo Playground dm_control_suite envs.


Usage
-----
    from dmc_partial_obs import wrap_partial_obs

    raw_env = registry.load(env_name, config=cfg, config_overrides=ovr)
    raw_env = wrap_partial_obs(raw_env, env_name, drop=("velocity",))

Two masking modes:
  * mode="drop"  -> obs dim shrinks (recommended; nothing wasted on constants)
  * mode="zero"  -> obs dim preserved, hidden entries set to 0.0
    (use when something downstream hard-codes the observation size)

"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping, Sequence, Tuple

import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground._src import mjx_env

OBS_LAYOUTS: Dict[str, Tuple[Tuple[str, int], ...]] = {
    "WalkerWalk": (("orientations", 14), ("height", 1), ("velocity", 9)),
    "WalkerRun": (("orientations", 14), ("height", 1), ("velocity", 9)),
    "WalkerStand": (("orientations", 14), ("height", 1), ("velocity", 9)),
    "FingerSpin": (("position", 4), ("velocity", 3), ("touch", 2)),
}

def detect_field_indices(
    env: Any,
    field: str = "qvel",
    n_probes: int = 8,
    n_warmup: int = 5,
    seed: int = 0,
    excite: bool = True,
    excite_scale: float = 2.0,
) -> np.ndarray:
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    candidates: Dict[int, set] | None = None
    informative: Dict[int, int] | None = None
    n_obs = 0

    for _ in range(n_probes):
        rng, key = jax.random.split(rng)
        state = reset_fn(key)

        if excite:
            rng, key = jax.random.split(rng)
            qvel = jax.random.uniform(
                key,
                state.data.qvel.shape,
                dtype=state.data.qvel.dtype,
                minval=-excite_scale,
                maxval=excite_scale,
            )
            state = state.replace(data=state.data.replace(qvel=qvel))

        for _ in range(n_warmup):
            rng, key = jax.random.split(rng)
            action = jax.random.uniform(
                key, (env.action_size,), minval=-1.0, maxval=1.0
            )
            state = step_fn(state, action)

        obs = state.obs["state"] if isinstance(state.obs, Mapping) else state.obs
        obs = np.asarray(obs).ravel()
        target = np.asarray(getattr(state.data, field)).ravel()

        if candidates is None:
            n_obs = int(obs.size)
            candidates = {j: set(range(n_obs)) for j in range(target.size)}
            informative = {j: 0 for j in range(target.size)}

        for j in range(target.size):
            candidates[j] &= {i for i in range(n_obs) if obs[i] == target[j]}
            if target[j] != 0.0:
                informative[j] += 1

    dead = [j for j, c in informative.items() if c == 0]
    if dead:
        raise ValueError(
            f"data.{field}[{dead}] was exactly 0.0 in every probe of "
            f"{type(env).__name__}, so it cannot be located by value: any "
            f"other zero column (an untouched contact sensor reports "
            f"log1p(0) == 0.0) is an equally good match. This is typical for "
            f"an unactuated DOF the probe actions never excite. Retry with "
            f"excite=True, declare the layout in OBS_LAYOUTS, or pass "
            f"--obs_keep_indices."
        )

    missing = [j for j, s in candidates.items() if not s]
    if missing:
        raise ValueError(
            f"data.{field}[{missing}] does not appear verbatim in the "
            f"observation of {type(env).__name__}. This env transforms or "
            f"omits it; inspect _get_obs and use --obs_keep_indices instead."
        )

    ambiguous = {j: sorted(s) for j, s in candidates.items() if len(s) > 1}
    if ambiguous:
        raise ValueError(
            f"Ambiguous match for data.{field} in {type(env).__name__}: "
            f"{ambiguous}. Retry with a larger n_probes, or declare the "
            f"layout in OBS_LAYOUTS."
        )

    return np.asarray(
        [candidates[j].pop() for j in sorted(candidates)], dtype=np.int32
    )


def cross_check_layout(
    env_name: str, detected: np.ndarray, field: str = "qvel"
) -> None:
    """Raises if a declared layout disagrees with what was detected."""
    if env_name not in OBS_LAYOUTS or field != "qvel":
        return
    layout = OBS_LAYOUTS[env_name]
    offset, expected = 0, None
    for name, size in layout:
        if name == "velocity":
            expected = np.arange(offset, offset + size, dtype=np.int32)
        offset += size
    if expected is None:
        return
    if not np.array_equal(expected, detected):
        raise ValueError(
            f"OBS_LAYOUTS['{env_name}'] puts velocity at {expected.tolist()} "
            f"but detection found {detected.tolist()}. The table is stale."
        )
    print(
        f"[partial-obs] layout for '{env_name}' verified against runtime "
        f"detection: velocity at {detected.tolist()}"
    )


def print_obs_layout(env_name: str) -> None:
    """Print `_get_obs` and the true obs dim so a layout can be filled in."""
    from mujoco_playground import registry

    env = registry.load(env_name)
    fn = getattr(type(env), "_get_obs", None)
    if fn is not None:
        print(inspect.getsource(fn))
    print(f"{env_name} observation_size = {env.observation_size}")
    print(f"declared layout           = {OBS_LAYOUTS.get(env_name)}")


def resolve_keep_indices(
    env_name: str,
    drop: Sequence[str],
    obs_dim: int,
) -> np.ndarray:
    """Map segment names to the indices that survive, or fail loudly."""
    if env_name not in OBS_LAYOUTS:
        raise KeyError(
            f"No observation layout registered for '{env_name}'. Run "
            f"dmc_partial_obs.print_obs_layout('{env_name}') and add it to "
            f"OBS_LAYOUTS, or pass explicit --obs_keep_indices."
        )
    layout = OBS_LAYOUTS[env_name]

    declared = sum(size for _, size in layout)
    if declared != obs_dim:
        raise ValueError(
            f"Layout for '{env_name}' declares {declared} dims but the env "
            f"reports {obs_dim}. The Playground observation changed; fix "
            f"OBS_LAYOUTS before training, otherwise you would be masking "
            f"arbitrary dimensions."
        )

    names = [name for name, _ in layout]
    unknown = [d for d in drop if d not in names]
    if unknown:
        raise ValueError(
            f"Unknown segment(s) {unknown} for '{env_name}'. Available: {names}"
        )

    keep, offset = [], 0
    for name, size in layout:
        if name not in drop:
            keep.extend(range(offset, offset + size))
        offset += size

    if not keep:
        raise ValueError(f"drop={tuple(drop)} would remove the entire observation.")
    return np.asarray(keep, dtype=np.int32)


class ObsMaskWrapper(mjx_env.MjxEnv):
    """Hides part of the observation vector of a Playground env.

    Delegates every non-observation attribute to the wrapped env, so brax
    training wrappers, the domain randomizer and `render` keep working.
    """

    def __init__(
        self,
        env: Any,
        keep_idx: np.ndarray,
        full_dim: int,
        mode: str = "drop",
        privileged: bool = False,
    ):
        if mode not in ("drop", "zero"):
            raise ValueError(f"mode must be 'drop' or 'zero', got {mode!r}")
        
        self._env = env
        self._mode = mode
        self._privileged = privileged
        self._full_dim = int(full_dim)
        self._keep_np = np.asarray(keep_idx, dtype=np.int32)

        
        if self._keep_np.size == 0:
            raise ValueError("keep_idx is empty; nothing would be observable.")
        if self._keep_np.min() < 0 or self._keep_np.max() >= self._full_dim:
            raise ValueError(
                f"keep_idx out of range [0, {self._full_dim}): "
                f"min={self._keep_np.min()}, max={self._keep_np.max()}."
            )
        if np.any(np.diff(self._keep_np) <= 0):
            raise ValueError("keep_idx must be strictly increasing and unique.")

        self._keep = jp.asarray(self._keep_np)
        if mode == "zero":
            m = np.zeros((self._full_dim,), dtype=np.float32)
            m[self._keep_np] = 1.0
            self._zero_mask = jp.asarray(m)

        
        self._obs_size_cache: Any = None

    def _apply(self, x: jax.Array) -> jax.Array:
        if self._mode == "drop":
            return x.at[..., self._keep].get(
                indices_are_sorted=True,
                unique_indices=True,
                mode="promise_in_bounds",
            )
        return x * self._zero_mask

    def _mask(self, obs):
        if isinstance(obs, Mapping):
            out = dict(obs)
            full = out["state"]
            out["state"] = self._apply(full)
            if self._privileged and "privileged_state" not in out:
                out["privileged_state"] = full
            return out
        masked = self._apply(obs)
        if self._privileged:
            return {"state": masked, "privileged_state": obs}
        return masked

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self._env.reset(rng)
        return state.replace(obs=self._mask(state.obs))

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self._env.step(state, action)
        return state.replace(obs=self._mask(state.obs))

    @property
    def observation_size(self):
        """Observation size, computed arithmetically and cached.

        """
        if self._obs_size_cache is not None:
            return self._obs_size_cache

        masked_dim = (
            len(self._keep_np) if self._mode == "drop" else self._full_dim
        )
        inner = self._env.observation_size

        if isinstance(inner, Mapping):
            out = dict(inner)
            lead = tuple(inner["state"][:-1])
            out["state"] = lead + (masked_dim,)
            if self._privileged and "privileged_state" not in inner:
                out["privileged_state"] = lead + (self._full_dim,)
        elif self._privileged:
            out = {
                "state": (masked_dim,),
                "privileged_state": (self._full_dim,),
            }
        else:
            out = masked_dim

        self._obs_size_cache = out
        return out


    @property
    def action_size(self):
        return self._env.action_size

    @property
    def unwrapped(self):
        return self._env.unwrapped

    @property
    def xml_path(self):
        return self._env.xml_path

    @property
    def mj_model(self):
        return self._env.mj_model

    @property
    def mjx_model(self):
        return self._env.mjx_model

    @property
    def dt(self):
        return self._env.dt

    @property
    def sim_dt(self):
        return self._env.sim_dt

    @property
    def n_substeps(self):
        return self._env.n_substeps

    def render(self, trajectory, *args, **kwargs):
        return self._env.render(trajectory, *args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            env = self.__dict__["_env"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(env, name)

    def __repr__(self):
        kept = len(self._keep_np)
        return (
            f"ObsMaskWrapper({self._env!r}, mode={self._mode}, "
            f"kept={kept}/{self._full_dim}, privileged={self._privileged})"
        )

def parse_index_spec(spec: str) -> np.ndarray:
    """'0:15,20' -> array([0..14, 20]). Escape hatch for unlisted envs."""
    idx = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            lo, hi = part.split(":")
            idx.extend(range(int(lo), int(hi)))
        else:
            idx.append(int(part))
    return np.asarray(sorted(set(idx)), dtype=np.int32)


def wrap_partial_obs(
    env: Any,
    env_name: str,
    drop: Sequence[str] = (),
    keep_indices: str | None = None,
    mode: str = "drop",
    privileged: bool = False,
    verify: bool = False,
) -> Any:
    """Wraps `env` unless nothing was asked for, in which case returns it.

    """
    if not drop and keep_indices is None:
        return env

    obs_size = env.observation_size
    if isinstance(obs_size, Mapping):
        full_dim = int(obs_size["state"][-1])
    else:
        full_dim = int(obs_size)

    detected = None

    if keep_indices is not None:
        keep = parse_index_spec(keep_indices)
        if keep.size == 0 or keep.max() >= full_dim:
            raise ValueError(
                f"--obs_keep_indices out of range for obs dim {full_dim}."
            )
        source = "explicit-indices"

    elif env_name in OBS_LAYOUTS:
        keep = resolve_keep_indices(env_name, drop, full_dim)
        source = "OBS_LAYOUTS"
        if verify:
            detected = detect_field_indices(env, field="qvel")
            cross_check_layout(env_name, detected)

    elif tuple(drop) == ("velocity",):
        detected = detect_field_indices(env, field="qvel")
        keep = np.asarray(
            [i for i in range(full_dim) if i not in set(detected.tolist())],
            dtype=np.int32,
        )
        source = "runtime-detection"
        print(f"[partial-obs] detected qvel at obs indices {detected.tolist()}")
        print(
            f"[partial-obs] consider adding a layout for '{env_name}' to "
            f"OBS_LAYOUTS so future runs skip detection entirely."
        )

    else:
        keep = resolve_keep_indices(env_name, drop, full_dim)
        source = "OBS_LAYOUTS"

    wrapped = ObsMaskWrapper(
        env, keep, full_dim, mode=mode, privileged=privileged
    )
    print(
        f"[partial-obs] {env_name}: {full_dim} -> {len(keep)} dims "
        f"(mode={mode}, dropped={tuple(drop) or 'explicit-indices'}, "
        f"privileged_critic={privileged}, source={source})"
    )
    return wrapped