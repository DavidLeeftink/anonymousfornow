#!/bin/bash -l
#SBATCH --job-name=mjx-eval
#SBATCH --partition=hawaii              
#SBATCH --array=1-10                  # Spawns 10 jobs (Seeds 1 to 10)
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                  
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=01:30:00               
#SBATCH -o logs/eval_%A_%a_output.log   
#SBATCH -e logs/eval_%A_%a_error.log               

# 1. Initialize Conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mj_playground2

# 2. Navigate to submit directory
cd $SLURM_SUBMIT_DIR
mkdir -p logs
DROPOUT_PROB=${1:-0.0}
ENV_NAME="H1JoystickGaitTracking"

SEED=$SLURM_ARRAY_TASK_ID

# Define your two experimental job folders

echo "=========================================================================="
echo "Starting Evaluation | Seed: $SEED | Host: $(hostname)"
echo "=========================================================================="

PROJECT_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../..")
export PYTHONPATH=$PROJECT_ROOT

# --- 1. Evaluate Policy A (Costate 0.0) ---
JOB_COSTATE_00="${ENV_NAME}_job72460"
PATH_COSTATE_00="${JOB_COSTATE_00}/seed${SEED}_costate0.0"
echo "[SLURM] Evaluating Baseline Policy (Costate 0.0): $PATH_COSTATE_00"

python -u evaluate_policies.py \
    --env_name="$ENV_NAME" \
    --load_run_path="$PATH_COSTATE_00" \
    --start_seed=1 \
    --end_seed=100 \
    --use_rnn=True \
    --costate_coeff=0. \
    --dropout_prob="$DROPOUT_PROB" \
    --render_video=False

# # --- 2. Evaluate Policy B (Costate 0.05) ---
JOB_COSTATE_05="${ENV_NAME}_job73698"
PATH_COSTATE_05="${JOB_COSTATE_05}/seed${SEED}_costate0.05"
echo "[SLURM] Evaluating Regularized Policy (Costate 0.05): $PATH_COSTATE_05"

python -u evaluate_policies.py \
    --env_name="$ENV_NAME" \
    --load_run_path="$PATH_COSTATE_05" \
    --start_seed=1 \
    --end_seed=100 \
    --use_rnn=True \
    --costate_coeff=0.05 \
    --dropout_prob="$DROPOUT_PROB" \
    --render_video=False


# --- 2. Evaluate Policy B (Costate 0.1) ---
JOB_COSTATE_05="${ENV_NAME}_job72459"
PATH_COSTATE_05="${JOB_COSTATE_05}/seed${SEED}_costate0.1"
echo "[SLURM] Evaluating Regularized Policy (Costate 0.1): $PATH_COSTATE_05"

python -u evaluate_policies.py \
    --env_name="$ENV_NAME" \
    --load_run_path="$PATH_COSTATE_05" \
    --start_seed=1 \
    --end_seed=100 \
    --use_rnn=True \
    --costate_coeff=0.1 \
    --dropout_prob="$DROPOUT_PROB" \
    --render_video=False


# # --- 2. Evaluate Policy B (Costate 0.15) ---
JOB_COSTATE_05="${ENV_NAME}_job73736"
PATH_COSTATE_05="${JOB_COSTATE_05}/seed${SEED}_costate0.15"
echo "[SLURM] Evaluating Regularized Policy (Costate 0.15): $PATH_COSTATE_05"

python -u evaluate_policies.py \
    --env_name="$ENV_NAME" \
    --load_run_path="$PATH_COSTATE_05" \
    --start_seed=1 \
    --end_seed=100 \
    --use_rnn=True \
    --costate_coeff=0.15 \
    --dropout_prob="$DROPOUT_PROB" \
    --render_video=False

## --- 2. Evaluate Policy B (Costate 0.2) ---
JOB_COSTATE_05="${ENV_NAME}_job73737"
PATH_COSTATE_05="${JOB_COSTATE_05}/seed${SEED}_costate0.2"
echo "[SLURM] Evaluating Regularized Policy (Costate 0.2): $PATH_COSTATE_05"

python -u evaluate_policies.py \
    --env_name="$ENV_NAME" \
    --load_run_path="$PATH_COSTATE_05" \
    --start_seed=1 \
    --end_seed=100 \
    --use_rnn=True \
    --costate_coeff=0.2 \
    --dropout_prob="$DROPOUT_PROB" \
    --render_video=False


echo "=========================================================================="
echo "Finished Seed $SEED Evaluation for all policies!"
echo "=========================================================================="