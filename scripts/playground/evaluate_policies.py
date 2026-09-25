# Copyright 2025 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0

import os
# Must be set before any mujoco imports
os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import imageio_ffmpeg
ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_bin)
os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]

import mediapy as media
from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import json
from ml_collections import config_dict
import mujoco
from mujoco_playground import registry
from mujoco_playground import wrapper_torch
from configs import locomotion_params
from configs import manipulation_params
from rsl_rl.runners import OnPolicyRunner
import torch
import warp as wp
import numpy as np

logging.set_verbosity(logging.WARNING)

# Evaluation CLI Flags
_DROPOUT_PROB = flags.DEFINE_float("dropout_prob", 0.0, "Probability of sensor measurement dropout (0.0 to 1.0).")
_ENV_NAME = flags.DEFINE_string("env_name", "H1JoystickGaitTracking", "Environment name.")
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_LOAD_RUN_PATH = flags.DEFINE_string("load_run_path", None, "Relative path inside logs/ to load from.")
_CHECKPOINT_NUM = flags.DEFINE_integer("checkpoint_num", -1, "Checkpoint number to load from (-1 for latest).")
_START_SEED = flags.DEFINE_integer("start_seed", 1, "Starting seed for evaluation range.")
_END_SEED = flags.DEFINE_integer("end_seed", 10, "Ending seed (inclusive) for evaluation range.")
_USE_RNN = flags.DEFINE_boolean("use_rnn", True, "Toggle between RNNModel and MLPModel.")
_COSTATE_COEFF = flags.DEFINE_float("costate_coeff", 0.0, "Coefficient for the costate loss.")
_CAMERA = flags.DEFINE_string("camera", "track", "Camera name to use for rendering.")
_DEVICE = flags.DEFINE_string("device", "cuda:0", "Device for evaluation.")
_RENDER_VIDEO = flags.DEFINE_boolean("render_video", False, "Whether to render and save an MP4 rollout per seed.")

## Sensor dropout function
def apply_full_sensor_dropout(obs_torch: torch.Tensor, p_drop: float) -> torch.Tensor:
    """Applies a Bernoulli trial to zero out sensor readings with probability p_drop."""
    if p_drop <= 0.0:
        return obs_torch
    
    # Generate 1s with prob (1 - p_drop) and 0s with prob p_drop
    mask = torch.bernoulli(torch.full_like(obs_torch, 1.0 - p_drop))
    return obs_torch * mask

def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in registry.manipulation._envs:
    return manipulation_params.rsl_rl_config(env_name)
  elif env_name in registry.locomotion._envs:
    return locomotion_params.rsl_rl_config(env_name)
  else:
    raise ValueError(f"No RL config found for {env_name}")

def adapt_playground_config(playground_config, use_rnn: bool = True, costate_coeff: float = 0.) -> dict:
    cfg = playground_config
    old_policy = cfg.pop("policy", {})
    old_algorithm = cfg.get("algorithm", {})
    
    is_normalized = cfg.get("empirical_normalization", True)
    cfg["empirical_normalization"] = is_normalized
    cfg["check_for_nan"] = cfg.get("check_for_nan", True)
    cfg["obs_groups"] = {"actor": ["state"], "critic": ["state"]}#privileged_
    
    old_algorithm["rnd_cfg"] = None
    old_algorithm["symmetry_cfg"] = None
    old_algorithm["costate_coeff"] = costate_coeff
    
    model_class = "RNNModel" if use_rnn else "MLPModel"
    activation_func = old_policy.get("activation", "elu")
    
    cfg["actor"] = {
        "class_name": model_class,
        "hidden_dims": old_policy.get("actor_hidden_dims", [512, 256, 128]),
        "activation": activation_func,
        "obs_normalization": is_normalized,
        "distribution_cfg": {
            "class_name": "GaussianDistribution", 
            "init_std": old_policy.get("init_noise_std", 1.0)
        }
    }
    cfg["critic"] = {
        "class_name": model_class,
        "hidden_dims": old_policy.get("critic_hidden_dims", [512, 256, 128]),
        "obs_normalization": is_normalized,
        "activation": activation_func
    }
    
    if use_rnn:
        rnn_spec = {
            "rnn_type": "gru",
            "rnn_hidden_dim": 256,
            "rnn_num_layers": 1,
        }
        cfg["actor"].update(rnn_spec)
        cfg["critic"].update(rnn_spec)
        
    return cfg

def find_checkpoint_path(run_dir: str, checkpoint_num: int = -1) -> str:
    search_dirs = [run_dir, os.path.join(run_dir, "checkpoints")]
    for d in search_dirs:
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if ("model" in f or f.endswith(".pt")) and not f.endswith(".json")]
            if files:
                files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]) if "_" in x and x.split("_")[1].split(".")[0].isdigit() else 0)
                target_file = files[checkpoint_num]
                return os.path.join(d, target_file)
    raise FileNotFoundError(f"Could not find any model checkpoint files in {run_dir} or {run_dir}/checkpoints")

def main(argv):
  del argv
  if not _LOAD_RUN_PATH.value:
    raise ValueError("You must specify --load_run_path pointing to the run folder inside logs/")

  device = _DEVICE.value
  device_rank = int(device.split(":")[-1]) if "cuda" in device else 0
  
  full_log_dir = os.path.abspath("logs")
  run_dir = os.path.join(full_log_dir, _LOAD_RUN_PATH.value)
  
  if not os.path.exists(run_dir):
      raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

  # 1. SETUP ENVIRONMENT & MODEL (Done exactly ONCE for maximum speed!)
  print(f"[EVAL] Initializing environment '{_ENV_NAME.value}' and compiling JAX kernels...", flush=True)
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env_cfg.impl = _IMPL.value
  raw_env = registry.load(_ENV_NAME.value, config=env_cfg)
  
  brax_env = wrapper_torch.RSLRLBraxWrapper(
      raw_env, 1, _START_SEED.value, env_cfg.episode_length, 1,
      render_callback=None, randomization_fn=None, device_rank=device_rank,
  )
  brax_env.cfg = env_cfg.to_dict()
  brax_env.device = device

  train_cfg = get_rl_config(_ENV_NAME.value)
  train_cfg.obs_groups = {"policy": ["state"], "critic": ["state"]}# privileged_
  rsl_rl_cfg = adapt_playground_config(train_cfg.to_dict(), use_rnn=_USE_RNN.value, costate_coeff=_COSTATE_COEFF.value)

  runner = OnPolicyRunner(brax_env, rsl_rl_cfg, log_dir=run_dir, device=device)
  resume_path = find_checkpoint_path(run_dir, _CHECKPOINT_NUM.value)
  print(f"[EVAL] Loading weights from: {resume_path}\n", flush=True)
  runner.load(resume_path)
  policy = runner.get_inference_policy(device=device)

  eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)
  is_dict_obs = isinstance(eval_env.observation_size, dict)

  # 2. SEQUENTIAL SEED EVALUATION LOOP
  npy_matrix_data = []
  json_summary_data = []

  print(f"================================================================")
  print(f"STARTING SEQUENTIAL EVALUATION | Seeds: {_START_SEED.value} to {_END_SEED.value}")
  print(f"================================================================", flush=True)

  for seed in range(_START_SEED.value, _END_SEED.value + 1):
      if _USE_RNN.value and hasattr(policy, "reset"):
        policy.reset()
      rng = jax.random.PRNGKey(seed)
      state = jit_reset(rng)
      
      rollout = [state] if _RENDER_VIDEO.value else []
      total_reward = 0.0
      steps_survived = 0

      obs = state.obs["state"] if is_dict_obs else state.obs
      obs_torch = wrapper_torch._jax_to_torch(obs)

      for step in range(env_cfg.episode_length):
          noisy_obs_torch = apply_full_sensor_dropout(obs_torch, p_drop=_DROPOUT_PROB.value)
          
          with torch.no_grad():
              actions = policy({"state": noisy_obs_torch})
              actions = torch.clip(actions, -1.0, 1.0)
              
          state = jit_step(state, wrapper_torch._torch_to_jax(actions.flatten()))
          if _RENDER_VIDEO.value:
              rollout.append(state)
            
          total_reward += float(state.reward)
          steps_survived += 1
            
          obs = state.obs["state"] if is_dict_obs else state.obs
          obs_torch = wrapper_torch._jax_to_torch(obs)
            
          if state.done:
              break

      survival_rate = steps_survived / env_cfg.episode_length
      mean_reward_per_step = total_reward / max(1, steps_survived)

      # Store row for .npy matrix: [Seed, Total Reward, Steps Survived, Survival Rate, Mean Reward/Step]
      npy_matrix_data.append([float(seed), total_reward, float(steps_survived), survival_rate, mean_reward_per_step])
      
      # Store detailed dict for JSON
      json_summary_data.append({
          "seed": seed,
          "total_reward": round(total_reward, 4),
          "steps_survived": steps_survived,
          "survival_rate": round(survival_rate, 4),
          "mean_reward_per_step": round(mean_reward_per_step, 4)
      })

      print(f"Seed {seed:02d} | Reward: {total_reward:8.2f} | Survived: {steps_survived:4d}/{env_cfg.episode_length} ({survival_rate*100:5.1f}%)", flush=True)

      # Optional Video Rendering per seed
      if _RENDER_VIDEO.value:
          scene_option = mujoco.MjvOption()
          scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
          scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
          scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

          render_every = 2
          fps = 1.0 / eval_env.dt / render_every
          traj = rollout[::render_every]
            
          try:
              frames = eval_env.render(traj, camera=_CAMERA.value, height=480, width=640, scene_option=scene_option)
          except ValueError:
              frames = eval_env.render(traj, camera=None, height=480, width=640, scene_option=scene_option)

          video_path = os.path.join(run_dir, f"eval_rollout_seed{seed}.mp4")
          media.write_video(video_path, frames, fps=fps)

  # 3. EXPORT RESULTS (.npy and .json)
  scores_matrix = np.array(npy_matrix_data, dtype=np.float32)
  npy_filename = f"eval_scores_seeds_{_START_SEED.value}_to_{_END_SEED.value}.npy"
  npy_path = os.path.join(run_dir, npy_filename)
  np.save(npy_path, scores_matrix)

  json_filename = f"eval_summary_seeds_{_START_SEED.value}_to_{_END_SEED.value}.json"
  json_path = os.path.join(run_dir, json_filename)
  with open(json_path, "w", encoding="utf-8") as f:
      json.dump({"metrics_by_seed": json_summary_data, "mean_total_reward": float(np.mean(scores_matrix[:, 1]))}, f, indent=4)

  print(f"\n================================================================")
  print(f"EVALUATION COMPLETE!")
  print(f"Saved NumPy Matrix:  {npy_path}")
  print(f"Saved JSON Summary:  {json_path}")
  print(f"Mean Reward across all {len(npy_matrix_data)} seeds: {np.mean(scores_matrix[:, 1]):.2f} ± {np.std(scores_matrix[:, 1]):.2f}")
  print(f"================================================================\n", flush=True)

if __name__ == "__main__":
  app.run(main)