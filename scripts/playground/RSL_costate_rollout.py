from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import tempfile
import numpy as np
import torch
from absl import app, flags

from rsl_rl.env.custom_envs import registry as custom_envs
from rsl_rl.runners import OnPolicyRunner

_RUN_DIR = flags.DEFINE_string("run_dir", None, "Training run directory.")
_DATA_DIR = flags.DEFINE_string("data_dir", None,
                                "Directory holding FHN_*_converged.npy. "
                                "Defaults to the parent of --run_dir's env folder.")
_CHECKPOINT = flags.DEFINE_string("checkpoint", None,
                                  "Checkpoint file. Defaults to the highest-numbered "
                                  "model_*.pt found under --run_dir.")
_ENV_NAME = flags.DEFINE_string("env_name", "FitzhughNagumo", "Custom env name.")
_DEVICE = flags.DEFINE_string("device", "cuda:0", "Device.")
_OUT_NAME = flags.DEFINE_string("out_name", "costate_rollout",
                                "Output subdirectory name inside --run_dir.")
_ALG_CLASS = flags.DEFINE_string("alg_class", "PPO",
                                 "algorithm.class_name, if absent from the saved config.")
_DIST_CLASS = flags.DEFINE_string("dist_class", "GaussianDistribution",
                                  "distribution_cfg.class_name, if absent.")


# ---------------------------------------------------------------- discovery


def find_checkpoint(run_dir: Path) -> Path:
    """Highest-numbered model_*.pt, searching the run dir and checkpoints/."""
    cands = list(run_dir.glob("model_*.pt")) + list(run_dir.glob("checkpoints/model_*.pt"))
    if not cands:
        raise FileNotFoundError(
            f"No model_*.pt under {run_dir} or {run_dir}/checkpoints. "
            f"Pass --checkpoint explicitly."
        )

    def num(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except ValueError:
            return -1

    return max(cands, key=num)


def load_resolved_config(run_dir: Path) -> dict:
    """The rsl_rl_cfg the run actually resolved to, written by the training script."""
    for rel in ("checkpoints/resolved_config.json", "resolved_config.json"):
        p = run_dir / rel
        if p.exists():
            blob = json.loads(p.read_text())
            cfg = blob.get("rsl_rl_cfg", blob)
            if "actor" not in cfg:
                raise ValueError(f"{p} has no 'actor' entry; cannot rebuild the model.")
            return cfg
    raise FileNotFoundError(
        f"No resolved_config.json under {run_dir}. It is written by "
        f"playground_RSL_training.py; without it the architecture is unknown."
    )


def infer_model_classes(ckpt: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"  could not inspect checkpoint ({exc}); falling back to the config")
        return out

    for role in ("actor", "critic"):
        sd = blob.get(f"{role}_state_dict")  # noqa: E501
        if not isinstance(sd, dict):
            continue
        has_rnn = any(k.startswith("rnn.") or ".rnn." in k for k in sd)
        out[role] = "RNNModel" if has_rnn else "MLPModel"
    return out


def repair_config(cfg: dict, ckpt: Path | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    from_ckpt = infer_model_classes(ckpt) if ckpt is not None else {}

    def restore(container: dict | None, key: str, value: str, where: str) -> None:
        if container is None or key in container:
            return
        container[key] = value
        print(f"  restored {where}.{key} = {value}")

    restore(cfg.setdefault("algorithm", {}), "class_name",
            _ALG_CLASS.value, "algorithm")

    for role in ("actor", "critic"):
        spec = cfg.get(role)
        if spec is None:
            raise ValueError(
                f"resolved_config.json has no '{role}' entry; the architecture "
                f"cannot be rebuilt."
            )
        if "class_name" not in spec:
            if role in from_ckpt:
                spec["class_name"] = from_ckpt[role]
                print(f"  restored {role}.class_name = {spec['class_name']}  "
                      f"(from checkpoint)")
            else:
                spec["class_name"] = "RNNModel" if "rnn_type" in spec else "MLPModel"
                print(f"  restored {role}.class_name = {spec['class_name']}  "
                      f"(from config)")
        elif role in from_ckpt and spec["class_name"] != from_ckpt[role]:
            print(f"  WARNING: config says {role} is {spec['class_name']} but the "
                  f"checkpoint looks like {from_ckpt[role]}")

        # Only the actor carries an action distribution.
        restore(spec.get("distribution_cfg"), "class_name",
                _DIST_CLASS.value, f"{role}.distribution_cfg")

        restore(spec, "readout", "linear" if role == "actor" else "mlp", role)

    # The runner must not resume; the checkpoint is loaded explicitly.
    cfg["resume"] = False
    cfg.setdefault("multi_gpu", None)
    return cfg


def remap_head_keys(ckpt: Path, runner) -> Path:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    models = {
        "actor": getattr(runner.alg, "_raw_actor", None),
        "critic": getattr(runner.alg, "_raw_critic", None),
    }
    if models["actor"] is None:
        models["actor"] = runner.alg.get_policy()

    changed = False
    for role, model in models.items():
        sd = blob.get(f"{role}_state_dict")
        if model is None or not isinstance(sd, dict):
            continue
        want = model.state_dict()
        renamed = []
        old_keys = [k for k in sd
                    if k.startswith("mlp.") and not k.startswith("mlp.head.")]
        for k in old_keys:
            if k in want:
                continue                      # already matches the model
            rest = k[len("mlp."):]            # e.g. "0.weight"
            idx, _, param = rest.partition(".")
            candidates = ["mlp.head.net." + rest]          # old MLP head
            if idx == "0":
                candidates.append("mlp.head.fc." + param)  # old linear head
            nk = next((c for c in candidates if c in want and c not in sd), None)
            if nk is None:
                continue
            if tuple(sd[k].shape) != tuple(want[nk].shape):
                raise ValueError(f"{role}: {k} {tuple(sd[k].shape)} does not fit "
                                 f"{nk} {tuple(want[nk].shape)}")
            sd[nk] = sd.pop(k)
            renamed.append(f"{k} -> {nk}")

        if renamed:
            changed = True
            print(f"  {role}: old head layout, remapped {len(renamed)} keys")
            for r in renamed:
                print(f"    {r}")

    if not changed:
        print("  checkpoint uses the current head layout")
        return ckpt
    fd, tmp = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(blob, tmp)
    return Path(tmp)


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Converged shooting solutions, canonicalized to [T, K, 2]."""
    s = data_dir / "FHN_states_converged.npy"
    l = data_dir / "FHN_costates_converged.npy"
    for p in (s, l):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Pass --data_dir.")
    X, L = np.load(s), np.load(l)
    if X.shape != L.shape:
        raise ValueError(f"shape mismatch: states {X.shape}, costates {L.shape}")

    d0 = np.abs(np.diff(X, axis=0)).mean()
    d1 = np.abs(np.diff(X, axis=1)).mean()
    if d1 < d0:
        X, L = np.swapaxes(X, 0, 1), np.swapaxes(L, 0, 1)
        print(f"  transposed to [T, K, 2]  (axis-0 diff {d0:.4g} > axis-1 {d1:.4g})")
    return np.ascontiguousarray(X), np.ascontiguousarray(L)


def find_rnn_module(actor):
    """The RNN wrapper that exposes costate_state(), or None."""
    if hasattr(actor, "rnn") and hasattr(actor.rnn, "costate_state"):
        return actor.rnn
    for m in actor.modules():
        if hasattr(m, "costate_state") and hasattr(m, "state_size"):
            return m
    return None


def extract_latent(actor, rnn_module) -> torch.Tensor:
    """Per-step hidden state h_t, [B, S]. Valid only right after a forward pass.

    Prefers costate_state(), which for the SSM cells resolves h rather than the
    read-out output C_t h_t + D x_t.
    """
    if rnn_module is not None:
        try:
            seq = rnn_module.costate_state()      # [1, B, S] in rollout mode
            return seq.reshape(-1, seq.shape[-1]).detach().clone()
        except RuntimeError:
            pass

    h = actor.get_hidden_state()
    if h is None:
        raise RuntimeError(
            "Could not obtain a hidden state. The actor exposes neither a usable "
            "costate_state() nor get_hidden_state()."
        )
    if isinstance(h, tuple):                       # LSTM: concatenate h and c
        h = torch.cat([t.reshape(t.shape[-2], -1) for t in h], dim=-1)
    else:
        h = h.reshape(h.shape[-2], -1)
    return h.detach().clone()


@torch.no_grad()
def rollout(env, policy, actor, rnn_module, horizon: int) -> dict[str, np.ndarray]:
    obs, _ = env.reset()
    policy.reset() if hasattr(policy, "reset") else actor.reset()

    states, latents, actions, rewards, discs = [], [], [], [], []
    for _ in range(horizon):
        z = env.get_states()
        act = policy(obs)                          # deterministic in eval mode
        h = extract_latent(actor, rnn_module)

        states.append(z.cpu())
        discs.append(env.spectral_discriminant(z).cpu())
        latents.append(h.cpu())
        actions.append(act.detach().cpu().clone())

        obs, rew, done, _ = env.step(act)
        rewards.append(rew.detach().cpu().clone())

    final = env.get_states().cpu()
    return {
        "states": torch.stack(states).numpy(),
        "latents": torch.stack(latents).numpy(),
        "actions": torch.stack(actions).numpy(),
        "rewards": torch.stack(rewards).numpy(),
        "discriminant": torch.stack(discs).numpy(),
        "final_state": final.numpy(),
    }


def objective(env, states: np.ndarray, actions: np.ndarray,
              final_state: np.ndarray) -> np.ndarray:
    """Undiscounted J of a logged trajectory, [K]. Comparable to the shooting cost."""
    running = (env.Q * (states ** 2).sum(-1) + env.R_cost * (actions ** 2).sum(-1))
    return running.sum(0) * env.dt + env.Qf * (final_state ** 2).sum(-1)


# --------------------------------------------------------------------- main


def main(argv):
    del argv
    if _RUN_DIR.value is None:
        raise ValueError("--run_dir is required.")

    run_dir = Path(_RUN_DIR.value).resolve()
    data_dir = Path(_DATA_DIR.value).resolve() if _DATA_DIR.value else run_dir.parent.parent
    device = _DEVICE.value

    print(f"run   {run_dir}")
    print(f"data  {data_dir}")

    ckpt = Path(_CHECKPOINT.value) if _CHECKPOINT.value else find_checkpoint(run_dir)
    print(f"ckpt  {ckpt.name}")

    cfg = repair_config(load_resolved_config(run_dir), ckpt)
    arch = cfg["actor"]
    print(f"arch  {arch.get('class_name')} / {arch.get('rnn_type')} "
          f"h={arch.get('rnn_hidden_dim')} l={arch.get('rnn_num_layers')} "
          f"readout={arch.get('readout')}")

    print("\nloading co-state dataset")
    true_X, true_L = load_dataset(data_dir)
    T, K, _ = true_X.shape
    print(f"  [T={T}, K={K}, 2]")

    init_states = torch.as_tensor(np.ascontiguousarray(true_X[0]), dtype=torch.float32)

    env_cfg = custom_envs.get_default_config(_ENV_NAME.value)
    if env_cfg.episode_length != T:
        print(f"  NOTE: env episode_length {env_cfg.episode_length} != dataset T {T}; "
              f"rolling for T steps.")
    env = custom_envs.load(_ENV_NAME.value, env_cfg, {}, num_envs=K, device=device)
    env.init_states = init_states.to(device)
    env.cfg = env_cfg.to_dict()
    env.device = device

    print("\nbuilding runner")
    runner = OnPolicyRunner(env, cfg, str(run_dir), device=device)
    load_path = remap_head_keys(ckpt, runner)
    try:
        runner.load(str(load_path))
    finally:
        if load_path != ckpt:
            load_path.unlink(missing_ok=True)
    runner.eval_mode() if hasattr(runner, "eval_mode") else runner.alg.eval_mode()

    policy = runner.get_inference_policy(device=device)
    actor = runner.alg.get_policy()
    rnn_module = find_rnn_module(actor)
    if rnn_module is None:
        print("  WARNING: no costate_state() found; falling back to get_hidden_state(). "
              "For SSM cells this may return the readout rather than h.")
    else:
        print(f"  latent source: costate_state(), state_size="
              f"{getattr(rnn_module, 'state_size', 'unknown')}")

    print(f"\nrolling {K} trajectories for {T} steps")
    out = rollout(env, policy, actor, rnn_module, horizon=T)
    print(f"  latents {out['latents'].shape}, states {out['states'].shape}")

    # Sanity: the rollout must start where the dataset starts.
    z0_err = np.abs(out["states"][0] - true_X[0]).max()
    print(f"  max |z_0 - z*_0| = {z0_err:.3e}  (should be ~0)")
    if z0_err > 1e-4:
        print("  WARNING: initial conditions do not match. Check env.init_states "
              "assignment and that num_envs == K.")

    achieved = objective(env, out["states"], out["actions"], out["final_state"])
    print(f"\n  achieved cost: mean {achieved.mean():.4f}  median "
          f"{np.median(achieved):.4f}")
    div = np.linalg.norm(out["states"] - true_X, axis=-1)
    print(f"  |z_t - z*_t|: median at t=0 {np.median(div[0]):.4f}, "
          f"t=T/2 {np.median(div[T // 2]):.4f}, t=T {np.median(div[-1]):.4f}")
    print("  (a per-timestep co-state comparison is only valid where this stays small)")

    out_dir = run_dir / Path(_OUT_NAME.value).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}
    arrays["true_states"] = true_X.astype(np.float32)
    arrays["true_costates"] = true_L.astype(np.float32)
    arrays["achieved_cost"] = achieved.astype(np.float32)
    arrays["state_divergence"] = div.astype(np.float32)

    for name, arr in arrays.items():
        np.save(out_dir / f"{name}.npy", arr)
    rnn_type_str = arch.get("rnn_type")
    costate_match = re.search(r"costate([0-9]+\.?[0-9]*)", str(run_dir))

    if costate_match and rnn_type_str is not None:
        costate_val = costate_match.group(1)
        rnn_type_str = f"{rnn_type_str}(CL{costate_val})"

    meta = {
        "rnn_type": rnn_type_str,
        "rnn_hidden_dim": arch.get("rnn_hidden_dim"),
        "rnn_num_layers": arch.get("rnn_num_layers"),
        "readout": arch.get("readout"),
        "model_class": arch.get("class_name"),
        "checkpoint": ckpt.name,
        "run_dir": str(run_dir),
        "env_name": _ENV_NAME.value,
        "env_cfg": env.cfg,
        "T": int(T),
        "K": int(K),
        "latent_dim": int(out["latents"].shape[-1]),
        "achieved_cost_mean": float(achieved.mean()),
        "achieved_cost_median": float(np.median(achieved)),
        "arrays": {k: list(v.shape) for k, v in arrays.items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # Read every array back. A file that cannot be reloaded is not a result.
    total = 0
    for name in arrays:
        chk = np.load(out_dir / f"{name}.npy")
        assert chk.shape == arrays[name].shape, f"{name}: shape changed on reload"
        total += (out_dir / f"{name}.npy").stat().st_size

    print(f"\nwrote {out_dir}  ({total / 1e6:.1f} MB, {len(arrays)} arrays, verified)")
    for name, arr in sorted(arrays.items()):
        print(f"    {name:20s} {arr.shape}")


if __name__ == "__main__":
    app.run(main)