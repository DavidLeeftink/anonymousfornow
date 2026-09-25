# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using RSL-RL for the specified environment."""

import os
# Must be set before any mujoco imports
os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

from datetime import datetime
import json

from absl import app
from absl import flags
from absl import logging
import jax
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from dmc_partial_obs import wrap_partial_obs
from configs import locomotion_params
from configs import manipulation_params
from configs import dm_control_suite_params
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env.custom_envs import registry as custom_envs
import torch
import warp as wp

import rsl_rl
print(f"!!! CURRENTLY USING RSL_RL FROM: {rsl_rl.__file__} !!!")

try:
  import wandb
except ImportError:
  wandb = None

logging.set_verbosity(logging.WARNING)

_ENV_NAME = flags.DEFINE_string("env_name", None, "Environment name.")
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax","warp"], "MJX implementation")
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string("playground_config_overrides", None, "Overrides.")
_LOAD_RUN_NAME = flags.DEFINE_string("load_run_name", None, "Run name to load from.")
_CHECKPOINT_NUM = flags.DEFINE_integer("checkpoint_num", -1, "Checkpoint number to load from.")
_PLAY_ONLY = flags.DEFINE_boolean("play_only", False, "If true, only play with the model.")
_USE_WANDB = flags.DEFINE_boolean("use_wandb", False, "Use Weights & Biases for logging.")
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name.")
_SEED = flags.DEFINE_integer("seed", 1, "Random seed.")
_NUM_ENVS = flags.DEFINE_integer("num_envs", int(4096), "Number of parallel envs.")
_USE_RNN = flags.DEFINE_boolean("use_rnn", False, "Toggle between RNNModel and MLPModel architectures.")
_DEVICE = flags.DEFINE_string("device", "cuda:0", "Device for training.")
_MULTI_GPU = flags.DEFINE_boolean("multi_gpu", False, "If true, use multi-GPU training.")
_CAMERA = flags.DEFINE_string("camera", "track", "Camera name to use for rendering.")
_WP_KERNEL_CACHE_DIR = flags.DEFINE_string("wp_kernel_cache_dir", "/tmp/wp_kernel_cache_playground", "WP cache.")
_READOUT = flags.DEFINE_enum("readout", "linear", ["linear", "mlp", "costate"],
                             "Actor readout. 'costate' is the G(y)h form of Def. 3.2.")
_COSTATE_COEFF = flags.DEFINE_float("costate_coeff", 0.0, "Coefficient for the costate loss.")

# --- Architecture / optimiser, exposed so sbatch fully specifies a run -------
_RNN_TYPE = flags.DEFINE_enum("rnn_type", "gru", ["gru", "lstm", "ncp-mamba", "mingru", "minlstm", "hopf", "complex-selective", "s4d", "lru"], "Recurrent cell type.")
_RNN_HIDDEN_DIM = flags.DEFINE_integer("rnn_hidden_dim", 256, "RNN hidden dimension.")
_RNN_NUM_LAYERS = flags.DEFINE_integer("rnn_num_layers", 1, "Number of RNN layers.")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", None, "PPO learning rate. Unset = env-tuned value.")
_LR_SCHEDULE = flags.DEFINE_enum("lr_schedule", None, ["fixed", "adaptive"], "LR schedule. Unset = config value.")
_MAX_ITERATIONS = flags.DEFINE_integer("max_iterations", None, "Policy updates. Unset = config value.")
_NUM_STEPS_PER_ENV = flags.DEFINE_integer("num_steps_per_env", None,
                                          "Rollout length. For recurrent policies this is ALSO the BPTT horizon.")


# --- Environment knob, as a first-class flag (no JSON quoting in sbatch) -----
_HISTORY_LEN = flags.DEFINE_integer("history_len", 1,
                                    "Length of the env's qvel / qpos_error history buffers.")
_OBS_DROP = flags.DEFINE_string("obs_drop", None,
                                "Comma-separated obs segments to hide, e.g. 'velocity'.")
_OBS_KEEP_INDICES = flags.DEFINE_string("obs_keep_indices", None,
                                        "Escape hatch: '0:15,20'. Overrides --obs_drop.")
_OBS_MASK_MODE = flags.DEFINE_enum("obs_mask_mode", "drop", ["drop", "zero"],
                                   "Shrink the obs vector or zero the hidden entries.")
_OBS_PRIVILEGED_CRITIC = flags.DEFINE_boolean("obs_privileged_critic", False,
                                              "Critic keeps the full state (asymmetric PPO).")


import mujoco.mjx.warp as mjxw
import warp.jax_experimental.ffi as _warp_ffi

if mjxw.types.GraphMode is int:
  mjxw.types.GraphMode = _warp_ffi.GraphMode
  
def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if custom_envs.is_custom(env_name):
    return custom_envs.rsl_rl_config(env_name)
  if env_name in registry.manipulation._envs:
    return manipulation_params.rsl_rl_config(env_name)
  elif env_name in registry.locomotion._envs:
    return locomotion_params.rsl_rl_config(env_name)
  elif env_name in registry.dm_control_suite._envs:
    return dm_control_suite_params.rsl_rl_config(env_name)
  else:
    raise ValueError(f"No RL config for {env_name}")


def adapt_playground_config(
    playground_config,
    use_rnn: bool = True,
    costate_coeff: float = 0.,
    rnn_type: str = "gru",
    readout: str = "linear",
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
    learning_rate=None,
    lr_schedule=None,
) -> dict:
    """
    Adapter that preserves MuJuCo Playground's robot-tuned hyperparameters
    while reshaping the configuration schema to match RSL-RL v5.
    """
    # 1. Convert the ml_collections.ConfigDict to a standard python dict
    cfg = playground_config#.to_dict()

    # 2. Extract the legacy components we need to migrate
    old_policy = cfg.pop("policy", {})
    old_algorithm = cfg.get("algorithm", {})

    is_normalized = cfg.get("empirical_normalization", True)
    cfg["empirical_normalization"] = is_normalized

    # 3. Inject new structural requirements for the v5 OnPolicyRunner
    cfg["check_for_nan"] = cfg.get("check_for_nan", True)
    cfg["obs_groups"] = {"actor": ["state"], "critic": ["state"]}#["privileged_state"]}

    # Ensure mandatory algorithm keys exist for v5 PPO
    old_algorithm["rnd_cfg"] = None
    old_algorithm["symmetry_cfg"] = None

    old_algorithm["costate_coeff"] = costate_coeff

    # Optimiser overrides from the command line (None = keep tuned value).
    if learning_rate is not None:
        old_algorithm["learning_rate"] = learning_rate
    if lr_schedule is not None:
        old_algorithm["schedule"] = lr_schedule

    # 4. Construct modern Model configurations using the robot's specific tuned dimensions
    model_class = "RNNModel" if use_rnn else "MLPModel"
    activation_func = old_policy.get("activation", "tanh")

    cfg["actor"] = {
        "class_name": model_class,
        "hidden_dims": tuple(old_policy.get("actor_hidden_dims", [512, 256, 128])),
        "readout": readout,
        "activation": activation_func,
        "obs_normalization": is_normalized,
        "squash_output": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": old_policy.get("init_noise_std", 1.0),
        },
    }

    cfg["critic"] = {
        "class_name": model_class,
        "hidden_dims": tuple(old_policy.get("critic_hidden_dims", [512, 256, 128])),
        "readout": "mlp",
        "obs_normalization": is_normalized,
        "activation": activation_func,
    }

    # 5. Inject recurrent parameters dynamically if RNN is enabled
    if use_rnn:
        rnn_spec = {
            "rnn_type": rnn_type,
            "rnn_hidden_dim": rnn_hidden_dim,
            "rnn_num_layers": rnn_num_layers,
        }
        cfg["actor"].update(rnn_spec)
        cfg["critic"].update(rnn_spec)

    return cfg

def _po(env):
  drop = tuple(s.strip() for s in _OBS_DROP.value.split(",")) if _OBS_DROP.value else ()
  return wrap_partial_obs(env, _ENV_NAME.value, drop=drop,
                          keep_indices=_OBS_KEEP_INDICES.value,
                          mode=_OBS_MASK_MODE.value,
                          privileged=_OBS_PRIVILEGED_CRITIC.value)

def build_env_overrides(env_cfg) -> dict:
  overrides = {}

  if custom_envs.is_custom(_ENV_NAME.value):
    if _HISTORY_LEN.value != 1:
      raise ValueError(
          f"--history_len={_HISTORY_LEN.value} requested but custom env "
          f"'{_ENV_NAME.value}' has no history buffer."
      )
    if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
      overrides.update(json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value))
    return overrides

  overrides["impl"] = _IMPL.value


  if "history_len" in env_cfg:
    overrides["history_len"] = _HISTORY_LEN.value
  elif _HISTORY_LEN.value != 1:
    raise ValueError(
        f"--history_len={_HISTORY_LEN.value} was requested but env "
        f"'{_ENV_NAME.value}' has no 'history_len' config key."
    )

  # Free-form JSON wins, so one-off experiments need no code change.
  if _PLAYGROUND_CONFIG_OVERRIDES.value is not None:
    overrides.update(json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value))

  return overrides


def main(argv):
  del argv

  wp.config.kernel_cache_dir = _WP_KERNEL_CACHE_DIR.value

  if _MULTI_GPU.value:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_rank = local_rank
    device = f"cuda:{local_rank}"
  else:
    device = _DEVICE.value
    device_rank = int(device.split(":")[-1]) if "cuda" in device else 0

  num_envs = 1 if _PLAY_ONLY.value else _NUM_ENVS.value

  is_custom_env = custom_envs.is_custom(_ENV_NAME.value)

  if is_custom_env:
    env_cfg = custom_envs.get_default_config(_ENV_NAME.value)
  else:
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    env_cfg.impl = _IMPL.value

  env_cfg_overrides = build_env_overrides(env_cfg)

  # --- DYNAMIC LOG DIRECTORY LOGIC ---
  if _PLAY_ONLY.value and _LOAD_RUN_NAME.value:
    # Playback: Use existing folder
    exp_name = _LOAD_RUN_NAME.value
    logdir = os.path.abspath(os.path.join("logs/", exp_name))
  else:
    # Training: Check if Bash provided a synchronized name
    if "EXP_NAME" in os.environ:
      exp_name = os.environ["EXP_NAME"]
    else:
      # Fallback for manual python runs
      now = datetime.now()
      timestamp = now.strftime("%Y%m%d-%H%M%S")
      exp_name = f"{_ENV_NAME.value}-{timestamp}-seed{_SEED.value}"
      if _SUFFIX.value is not None:
        exp_name += f"-{_SUFFIX.value}"
    logdir = os.path.abspath(os.path.join("logs/", exp_name))

  ckpt_path = os.path.join(logdir, "checkpoints")
  os.makedirs(ckpt_path, exist_ok=True)

  if _USE_WANDB.value and not _PLAY_ONLY.value and wandb is not None:
    wandb.tensorboard.patch(root_logdir=logdir)
    wandb.init(project="mjxrl", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  if not _PLAY_ONLY.value:
    with open(os.path.join(ckpt_path, "config.json"), "w", encoding="utf-8") as fp:
      json.dump(env_cfg.to_dict(), fp, indent=4)


  render_trajectory = []

  def render_callback(_, state):
    render_trajectory.append(state)

  if is_custom_env:
    # Plain torch VecEnv: no brax wrapper, no MJX model, no randomizer.
    train_env = custom_envs.load(
        _ENV_NAME.value, env_cfg, env_cfg_overrides,
        num_envs=num_envs, device=device,
    )
    train_env.cfg = env_cfg.to_dict()
    train_env.device = device
    obs_size = train_env.observation_size
    action_size = train_env.action_size
  else:
    randomizer = registry.get_domain_randomizer(_ENV_NAME.value)
    raw_env = registry.load(_ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides)
    raw_env = _po(raw_env)                     # <-- add
    train_env = wrapper_torch.RSLRLBraxWrapper(
        raw_env, num_envs, _SEED.value, env_cfg.episode_length, 1,
        render_callback=render_callback, randomization_fn=randomizer, device_rank=device_rank,
    )
    if not hasattr(train_env, "cfg"):
        train_env.cfg = env_cfg.to_dict()
    if not hasattr(train_env, "device"):
        train_env.device = device
    obs_size = raw_env.observation_size
    action_size = raw_env.action_size

  train_cfg = get_rl_config(_ENV_NAME.value)
  if isinstance(obs_size, dict) and "privileged_state" in obs_size:
    train_cfg.obs_groups = {"policy": ["state"], "critic": ["privileged_state"]} # ["privileged_state"]
  else:
    train_cfg.obs_groups = {"policy": ["state"], "critic": ["state"]}

  train_cfg.seed = _SEED.value
  train_cfg.run_name = exp_name
  train_cfg.resume = _LOAD_RUN_NAME.value is not None
  train_cfg.load_run = _LOAD_RUN_NAME.value if _LOAD_RUN_NAME.value else "-1"
  train_cfg.checkpoint = _CHECKPOINT_NUM.value

  if _MAX_ITERATIONS.value is not None:
    train_cfg.max_iterations = _MAX_ITERATIONS.value
  if _NUM_STEPS_PER_ENV.value is not None:
    train_cfg.num_steps_per_env = _NUM_STEPS_PER_ENV.value

  train_cfg_dict = train_cfg.to_dict()
  rsl_rl_cfg = adapt_playground_config(
      train_cfg_dict,
      use_rnn=_USE_RNN.value,
      costate_coeff=_COSTATE_COEFF.value,
      rnn_type=_RNN_TYPE.value,
      readout=_READOUT.value,
      rnn_hidden_dim=_RNN_HIDDEN_DIM.value,
      rnn_num_layers=_RNN_NUM_LAYERS.value,
      learning_rate=_LEARNING_RATE.value,
      lr_schedule=_LR_SCHEDULE.value,
  )

  runner = OnPolicyRunner(train_env, rsl_rl_cfg, logdir, device=device)

  # --- Resolved-configuration report -----------------------------------------
  algo = rsl_rl_cfg.get("algorithm", {})
  summary = {
      "env_name": _ENV_NAME.value,
      "seed": _SEED.value,
      "num_envs": num_envs,
      "observation_size": obs_size,
      "action_size": action_size,
      "use_rnn": _USE_RNN.value,
      "history_len": env_cfg_overrides.get("history_len"),
      "learning_rate": algo.get("learning_rate"),
      "lr_schedule": algo.get("schedule"),
      "num_steps_per_env": rsl_rl_cfg.get("num_steps_per_env"),
      "max_iterations": rsl_rl_cfg.get("max_iterations"),
      "ctrl_dt": float(env_cfg.ctrl_dt),
      "actor": rsl_rl_cfg.get("actor"),
      "critic": rsl_rl_cfg.get("critic"),
      "costate_coeff": _COSTATE_COEFF.value,
  }
  try:
    actor = runner.alg.get_policy()
    summary["actor_num_params"] = sum(p.numel() for p in actor.parameters())
  except Exception as exc:  # pylint: disable=broad-except
    summary["actor_num_params"] = f"unavailable: {exc}"

  if _USE_RNN.value:
    steps = rsl_rl_cfg.get("num_steps_per_env") or 0
    summary["bptt_horizon_steps"] = steps
    summary["bptt_horizon_seconds"] = steps * float(env_cfg.ctrl_dt)

  print("\n===================== RESOLVED CONFIG =====================")
  print(json.dumps(summary, indent=2, default=str))
  print("===========================================================\n")
  with open(os.path.join(ckpt_path, "resolved_config.json"), "w", encoding="utf-8") as fp:
    json.dump({"summary": summary, "rsl_rl_cfg": rsl_rl_cfg}, fp, indent=2, default=str)

  if train_cfg.resume:
    resume_path = wrapper_torch.get_load_path("logs/", load_run=train_cfg.load_run, checkpoint=train_cfg.checkpoint)
    runner.load(resume_path)

  # # --- TRAIN AND FALL THROUGH TO EVALUATION ---
  if not _PLAY_ONLY.value:
    runner.learn(num_learning_iterations=train_cfg.max_iterations, init_at_random_ep_len=False)
    print("Done training. Proceeding to final rollout video generation...")

  if is_custom_env:
    print("Custom env: skipping MuJoCo rollout and rendering.")
    return
    
  policy = runner.get_inference_policy(device=device)

  policy.reset()
  eval_env = _po(registry.load(_ENV_NAME.value, config=env_cfg, config_overrides=env_cfg_overrides))
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  rng = jax.random.PRNGKey(_SEED.value)
  state = jit_reset(rng)
  rollout = [state]

  is_dict_obs = isinstance(eval_env.observation_size, dict)
  obs = state.obs["state"] if is_dict_obs else state.obs
  obs_torch = wrapper_torch._jax_to_torch(obs)

  for _ in range(env_cfg.episode_length):
    with torch.no_grad():
      actions = policy({"state": obs_torch})
      actions = torch.clip(actions, -1.0, 1.0)
    state = jit_step(state, wrapper_torch._torch_to_jax(actions.flatten()))
    rollout.append(state)
    obs = state.obs["state"] if is_dict_obs else state.obs
    obs_torch = wrapper_torch._jax_to_torch(obs)
    if state.done:
      break

  # --- VIDEO RENDERING & EXPORT ---
  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  render_every = 2
  base_env = eval_env
  fps = 1.0 / base_env.dt / render_every
  traj = rollout[::render_every]

  # Safe Camera Fallback
  try:
      frames = eval_env.render(traj, camera=_CAMERA.value, height=480, width=640, scene_option=scene_option)
  except ValueError:
      print(f"Warning: Camera '{_CAMERA.value}' not found. Falling back to default free camera.")
      frames = eval_env.render(traj, camera=None, height=480, width=640, scene_option=scene_option)

  video_path = os.path.join(logdir, "rollout.mp4")
  media.write_video(video_path, frames, fps=fps)
  print(f"Rollout video saved successfully at: {video_path}")

def run():
  app.run(main)

if __name__ == "__main__":
  run()
