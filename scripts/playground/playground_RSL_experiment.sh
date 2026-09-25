#!/bin/bash -l
#SBATCH --job-name=mjx-ppo
#SBATCH --partition=oahu 
#SBATCH --array=1-10
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=11GB
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%A_%a_output.log
#SBATCH -e logs/%x_%A_%a_error.log

# example:
#   sbatch playground_RSL_experiment.sh --env=H1JoystickGaitTracking --rnn=True --rnn_type=gru --costate=0. --rnn_hidden=256 --lr=3e-4
#
#   sbatch playground_RSL_experiment.sh --env=H1JoystickGaitTracking --rnn=False --history_len=8 --lr=5e-4
# 
# - Continuing training from a policy:
#   sbatch --array=1-5 playground_RSL_experiment.sh --env=H1JoystickGaitTracking \
#       --rnn=True --history_len=1 --load_run=H1JoystickGaitTracking/mlp_hist8/job12345_seedSEED --max_iter=5000
#
# Args are --key=value, order-independent, all optional except --env.

set -o pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mj_playground2

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

# ---------------------------------------------------------------- defaults --
ENV_NAME=""
RNN_USE="False"
RNN_TYPE="gru"
RNN_HIDDEN=256
RNN_LAYERS=1
LR=""                 # empty  ->  keep the env-tuned value from locomotion_params
LR_SCHEDULE=""        # empty  ->  keep config value ("fixed" by default)
READOUT="linear"
COSTATE=0.0
HISTORY_LEN=1
LOAD_RUN=""            # empty -> fresh run. May contain literal token SEED, replaced
                       # below with the array task id, so one sbatch call can resume
                       # every seed from its own checkpoint.
CHECKPOINT_NUM=""      # empty -> keep python default (-1 = latest checkpoint)
PLAY_ONLY="False"      # True -> load checkpoint and roll out only, no further training
NUM_ENVS=4096
MAX_ITER=""           # empty  ->  keep config value (100000 for H1 -- set this!)
NUM_STEPS=""          # rollout length; ALSO the BPTT horizon for recurrent runs
IMPL="jax"
EXTRA_JSON=""         # free-form playground config overrides, e.g. '{"action_scale":0.5}'

usage() {
  cat <<'EOF'
Usage: sbatch playground_RSL_experiment.sh --env=NAME [options]

  --env=NAME               registry env name                     (required)
  --rnn=True|False         RNNModel vs MLPModel                   (False)
  --rnn_type=NAME      recurrent cell                         (gru)
  --rnn_hidden=INT         RNN hidden dim                         (256)
  --rnn_layers=INT         RNN layers                             (1)
  --lr=FLOAT               PPO learning rate            (unset = env-tuned)
  --lr_schedule=fixed|adaptive                          (unset = config)
  --costate=FLOAT          costate loss coefficient               (0.0)
  --readout=NAME           actor readout (linear, mlp, costate)   (linear)
  --history_len=INT        qvel/qpos_error history length         (1)
  --load_run=NAME          resume from logs/NAME                  (unset = fresh run)
                            Include the literal token SEED to resume each array
                            task from its own seed's checkpoint, e.g.
                            --load_run=H1Joystick/mlp_hist8/job123_seedSEED
  --checkpoint=INT         checkpoint number to resume from       (unset = latest)
  --play_only=True|False   load checkpoint, roll out only, no training (False)
  --num_envs=INT                                                  (4096)
  --max_iter=INT           policy updates               (unset = config)
  --num_steps=INT          num_steps_per_env = BPTT horizon (unset = config)
  --impl=jax|warp                                                 (jax)
  --json='{...}'           extra playground config overrides      (none)
EOF
}

for arg in "$@"; do
  case $arg in
    --env=*)         ENV_NAME="${arg#*=}" ;;
    --rnn=*)         RNN_USE="${arg#*=}" ;;
    --rnn_type=*)    RNN_TYPE="${arg#*=}" ;;
    --rnn_hidden=*)  RNN_HIDDEN="${arg#*=}" ;;
    --rnn_layers=*)  RNN_LAYERS="${arg#*=}" ;;
    --lr=*)          LR="${arg#*=}" ;;
    --lr_schedule=*) LR_SCHEDULE="${arg#*=}" ;;
    --costate=*)     COSTATE="${arg#*=}" ;;
    --readout=*)     READOUT="${arg#*=}" ;;
    --history_len=*) HISTORY_LEN="${arg#*=}" ;;
    --load_run=*)    LOAD_RUN="${arg#*=}" ;;
    --checkpoint=*)  CHECKPOINT_NUM="${arg#*=}" ;;
    --play_only=*)   PLAY_ONLY="${arg#*=}" ;;
    --num_envs=*)    NUM_ENVS="${arg#*=}" ;;
    --max_iter=*)    MAX_ITER="${arg#*=}" ;;
    --num_steps=*)   NUM_STEPS="${arg#*=}" ;;
    --impl=*)        IMPL="${arg#*=}" ;;
    --json=*)        EXTRA_JSON="${arg#*=}" ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "ERROR: unknown argument '$arg'"; usage; exit 1 ;;
  esac
done

if [ "$RNN_USE" != "True" ] && [ "$RNN_USE" != "true" ]; then
  if [ "$READOUT" != "linear" ] || [ "$COSTATE" != "0.0" ]; then
    echo "ERROR: --readout and --costate require --rnn=True."
    exit 1
  fi
fi

if [ -z "$ENV_NAME" ]; then
  echo "ERROR: --env is required."
  usage
  exit 1
fi

# ------------------------------------------------------------ run identity --
# Every knob that changes the experiment goes in the directory name. Two arms
# of the same sweep must never differ only by job id.
if [ "$RNN_USE" = "True" ] || [ "$RNN_USE" = "true" ]; then
  ARCH="rnn-${RNN_TYPE}-h${RNN_HIDDEN}-l${RNN_LAYERS}"
else
  ARCH="mlp"
fi
ARCH="${ARCH}_hist${HISTORY_LEN}"
[ -n "$LR" ] && ARCH="${ARCH}_lr${LR}"
[ "$COSTATE" != "0.0" ] && ARCH="${ARCH}_costate${COSTATE}"
[ "$READOUT" != "linear" ] && ARCH="${ARCH}_ro${READOUT}"

JOBID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
TASKID="${SLURM_ARRAY_TASK_ID:-0}"


RESOLVED_LOAD_RUN="${LOAD_RUN//SEED/$TASKID}"

export EXP_NAME="${ENV_NAME}/${ARCH}/job${JOBID}_seed${TASKID}"

echo "=========================================================="
echo " env          : $ENV_NAME"
echo " arch         : $ARCH"
echo " seed         : $TASKID"
echo " lr           : ${LR:-<env-tuned>}   schedule: ${LR_SCHEDULE:-<config>}"
echo " readout      : $READOUT"
echo " history_len  : $HISTORY_LEN"
echo " load_run     : ${RESOLVED_LOAD_RUN:-<fresh run>}   checkpoint: ${CHECKPOINT_NUM:-<latest>}   play_only: $PLAY_ONLY"
echo " num_envs     : $NUM_ENVS"
echo " max_iter     : ${MAX_ITER:-<config>}   num_steps: ${NUM_STEPS:-<config>}"
echo " logdir       : logs/$EXP_NAME"
echo "=========================================================="

# ------------------------------------------------------ optional overrides --
EXTRA=""
[ -n "$LR" ]          && EXTRA="$EXTRA --learning_rate=$LR"
[ -n "$LR_SCHEDULE" ] && EXTRA="$EXTRA --lr_schedule=$LR_SCHEDULE"
[ -n "$MAX_ITER" ]    && EXTRA="$EXTRA --max_iterations=$MAX_ITER"
[ -n "$NUM_STEPS" ]   && EXTRA="$EXTRA --num_steps_per_env=$NUM_STEPS"
[ -n "$RESOLVED_LOAD_RUN" ] && EXTRA="$EXTRA --load_run_name=$RESOLVED_LOAD_RUN"
[ -n "$CHECKPOINT_NUM" ]    && EXTRA="$EXTRA --checkpoint_num=$CHECKPOINT_NUM"
if [ "$PLAY_ONLY" = "True" ] || [ "$PLAY_ONLY" = "true" ]; then
  EXTRA="$EXTRA --play_only=True"
fi

PROJECT_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../..")
export PYTHONPATH=$PROJECT_ROOT

# Provenance: record which code produced this run.
mkdir -p "logs/${EXP_NAME}"
git -C "$SLURM_SUBMIT_DIR" rev-parse HEAD > "logs/${EXP_NAME}/git_sha.txt" 2>/dev/null
git -C "$SLURM_SUBMIT_DIR" diff > "logs/${EXP_NAME}/git_diff.patch" 2>/dev/null
cp playground_RSL_training.py "logs/${EXP_NAME}/" 2>/dev/null

if [ -n "$EXTRA_JSON" ]; then
  python -u playground_RSL_training.py \
      --env_name="$ENV_NAME" --impl="$IMPL" --seed="$TASKID" \
      --num_envs="$NUM_ENVS" --use_wandb=False \
      --use_rnn="$RNN_USE" --rnn_type="$RNN_TYPE" \
      --rnn_hidden_dim="$RNN_HIDDEN" --rnn_num_layers="$RNN_LAYERS" \
      --costate_coeff="$COSTATE" --history_len="$HISTORY_LEN" \
      --readout="$READOUT" \
      --playground_config_overrides="$EXTRA_JSON" \
      $EXTRA
else
  python -u playground_RSL_training.py \
      --env_name="$ENV_NAME" --impl="$IMPL" --seed="$TASKID" \
      --num_envs="$NUM_ENVS" --use_wandb=False \
      --use_rnn="$RNN_USE" --rnn_type="$RNN_TYPE" \
      --rnn_hidden_dim="$RNN_HIDDEN" --rnn_num_layers="$RNN_LAYERS" \
      --readout="$READOUT" \
      --costate_coeff="$COSTATE" --history_len="$HISTORY_LEN" \
      $EXTRA
fi
PY_STATUS=$?

# ------------------------------------------------------------- archive logs --
CURRENT_OUT_LOG="logs/${SLURM_JOB_NAME}_${JOBID}_${TASKID}_output.log"
CURRENT_ERR_LOG="logs/${SLURM_JOB_NAME}_${JOBID}_${TASKID}_error.log"
sleep 2
mkdir -p "logs/${EXP_NAME}"
[ -f "$CURRENT_OUT_LOG" ] && mv "$CURRENT_OUT_LOG" "logs/${EXP_NAME}/slurm_output.log"
[ -f "$CURRENT_ERR_LOG" ] && mv "$CURRENT_ERR_LOG" "logs/${EXP_NAME}/slurm_error.log"

echo "Slurm log files archived in logs/${EXP_NAME}/ (python exit $PY_STATUS)"
exit $PY_STATUS