#!/bin/bash -l
#SBATCH --job-name=warp-ppo
#SBATCH --partition=lanai,molokai
#SBATCH --array=1-10
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --time=36:00:00
#SBATCH -o logs/%x_%A_%a_output.log
#SBATCH -e logs/%x_%A_%a_error.log

# example:
#   sbatch playgroundManipulation_RSL_experiment.sh --env=CheetahRun --rnn=True --rnn_type=gru --costate=0.0 --rnn_hidden=256 --lr=1e-4 --obs_drop=velocity --readout=linear
#
#   sbatch playgroundManipulation_RSL_experiment.sh --env=WalkerWalk --rnn=False --history_len=8 --lr=5e-4
#
# - Partially observable dm_control (velocities hidden, recurrent policy):
#   sbatch --array=1-5 playgroundManipulation_RSL_experiment.sh --env=WalkerWalk \
#       --rnn=True --rnn_type=gru --obs_drop=velocity
#
# - Same task, fully observed baseline (omit --obs_drop):
#   sbatch --array=1-5 playgroundManipulation_RSL_experiment.sh --env=WalkerWalk --rnn=True
#
# - Continuing training from a policy:
#   sbatch --array=1-5 playgroundManipulation_RSL_experiment.sh --env=H1JoystickGaitTracking \
#       --rnn=True --history_len=1 --load_run=H1JoystickGaitTracking/mlp_hist8/job12345_seedSEED --max_iter=5000
#
# Args are --key=value, order-independent, all optional except --env.

set -o pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mjplayground2_manip

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
NUM_ENVS=1024
MAX_ITER=""           # empty  ->  keep config value (100000 for H1 -- set this!)
NUM_STEPS=""          # rollout length; ALSO the BPTT horizon for recurrent runs
IMPL="jax"
EXTRA_JSON=""         # free-form playground config overrides, e.g. '{"action_scale":0.5}'

# --- Partial observability (dm_control_suite) -------------------------------
# All empty/False -> fully observed, byte-identical to the pre-POMDP behaviour.
OBS_DROP=""            # e.g. "velocity" -> hides that segment of the flat obs
OBS_KEEP_INDICES=""    # escape hatch, e.g. "0:15" -- overrides --obs_drop
OBS_MASK_MODE="drop"   # drop = obs shrinks; zero = dim kept, entries zeroed
OBS_PRIV_CRITIC="False"  # critic keeps the full state (asymmetric PPO)

usage() {
  cat <<'EOF'
Usage: sbatch playground_RSL_experiment.sh --env=NAME [options]

  --env=NAME               registry env name                     (required)
  --rnn=True|False         RNNModel vs MLPModel                   (False)
  --rnn_type=NAME          recurrent cell                         (gru)
  --rnn_hidden=INT         RNN hidden dim                         (256)
  --rnn_layers=INT         RNN layers                             (1)
  --lr=FLOAT               PPO learning rate            (unset = env-tuned)
  --lr_schedule=fixed|adaptive                          (unset = config)
  --costate=FLOAT          costate loss coefficient               (0.0)
  --readout=NAME           readout layer (linear, MLP, costate) default linear
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

 Partial observability (dm_control_suite envs):
  --obs_drop=SEGS          comma-separated obs segments to hide   (none)
                            e.g. --obs_drop=velocity  (WalkerWalk: 24 -> 15)
  --obs_keep_indices=SPEC  explicit indices, e.g. '0:15,20'. Overrides --obs_drop.
  --obs_mask_mode=drop|zero  shrink the obs vector, or zero hidden entries (drop)
  --obs_privileged_critic=True|False  critic sees the full state  (False)
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
    --obs_drop=*)            OBS_DROP="${arg#*=}" ;;
    --obs_keep_indices=*)    OBS_KEEP_INDICES="${arg#*=}" ;;
    --obs_mask_mode=*)       OBS_MASK_MODE="${arg#*=}" ;;
    --obs_privileged_critic=*) OBS_PRIV_CRITIC="${arg#*=}" ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "ERROR: unknown argument '$arg'"; usage; exit 1 ;;
  esac
done

if [ -z "$ENV_NAME" ]; then
  echo "ERROR: --env is required."
  usage
  exit 1
fi

case "$OBS_MASK_MODE" in
  drop|zero) ;;
  *) echo "ERROR: --obs_mask_mode must be 'drop' or 'zero', got '$OBS_MASK_MODE'."; exit 1 ;;
esac

if [ -n "$OBS_DROP" ] && [ -n "$OBS_KEEP_INDICES" ]; then
  echo "ERROR: pass --obs_drop OR --obs_keep_indices, not both."
  exit 1
fi
if { [ "$OBS_PRIV_CRITIC" = "True" ] || [ "$OBS_PRIV_CRITIC" = "true" ]; } \
   && [ -z "$OBS_DROP" ] && [ -z "$OBS_KEEP_INDICES" ]; then
  echo "ERROR: --obs_privileged_critic=True is meaningless without masking;"
  echo "       the critic would already see the same full state as the actor."
  exit 1
fi

# ------------------------------------------------------------ run identity --
if [ "$RNN_USE" = "True" ] || [ "$RNN_USE" = "true" ]; then
  ARCH="rnn-${RNN_TYPE}-h${RNN_HIDDEN}-l${RNN_LAYERS}"
else
  ARCH="mlp"
fi
ARCH="${ARCH}_hist${HISTORY_LEN}"

OBS_TAG=""
if [ -n "$OBS_KEEP_INDICES" ]; then
  OBS_TAG="po-idx${OBS_KEEP_INDICES//[:,]/-}"
elif [ -n "$OBS_DROP" ]; then
  OBS_TAG="po-no${OBS_DROP//,/-}"
fi
if [ -n "$OBS_TAG" ]; then
  [ "$OBS_MASK_MODE" = "zero" ] && OBS_TAG="${OBS_TAG}-zero"
  if [ "$OBS_PRIV_CRITIC" = "True" ] || [ "$OBS_PRIV_CRITIC" = "true" ]; then
    OBS_TAG="${OBS_TAG}-privcritic"
  fi
  ARCH="${ARCH}_${OBS_TAG}"
fi

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
echo " observability: ${OBS_TAG:-<fully observed>}"
echo "                drop=${OBS_DROP:-<none>} keep_idx=${OBS_KEEP_INDICES:-<none>} mode=$OBS_MASK_MODE priv_critic=$OBS_PRIV_CRITIC"
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

if [ -n "$OBS_DROP" ] || [ -n "$OBS_KEEP_INDICES" ]; then
  [ -n "$OBS_DROP" ]         && EXTRA="$EXTRA --obs_drop=$OBS_DROP"
  [ -n "$OBS_KEEP_INDICES" ] && EXTRA="$EXTRA --obs_keep_indices=$OBS_KEEP_INDICES"
  EXTRA="$EXTRA --obs_mask_mode=$OBS_MASK_MODE"
  if [ "$OBS_PRIV_CRITIC" = "True" ] || [ "$OBS_PRIV_CRITIC" = "true" ]; then
    EXTRA="$EXTRA --obs_privileged_critic=True"
  fi
fi

PROJECT_ROOT=$(realpath "$SLURM_SUBMIT_DIR/../..")
export PYTHONPATH=$PROJECT_ROOT

# Provenance: record which code produced this run.
mkdir -p "logs/${EXP_NAME}"
git -C "$SLURM_SUBMIT_DIR" rev-parse HEAD > "logs/${EXP_NAME}/git_sha.txt" 2>/dev/null
git -C "$SLURM_SUBMIT_DIR" diff > "logs/${EXP_NAME}/git_diff.patch" 2>/dev/null
cp playgroundManipulation_RSL_training.py "logs/${EXP_NAME}/" 2>/dev/null


cp dmc_partial_obs.py "logs/${EXP_NAME}/" 2>/dev/null

if [ -n "$EXTRA_JSON" ]; then
  python -u playgroundManipulation_RSL_training.py \
      --env_name="$ENV_NAME" --impl="$IMPL" --seed="$TASKID" \
      --num_envs="$NUM_ENVS" --use_wandb=False \
      --use_rnn="$RNN_USE" --rnn_type="$RNN_TYPE" \
      --rnn_hidden_dim="$RNN_HIDDEN" --rnn_num_layers="$RNN_LAYERS" \
      --costate_coeff="$COSTATE" --history_len="$HISTORY_LEN" \
      --playground_config_overrides="$EXTRA_JSON" \
      --readout="$READOUT" \
      $EXTRA
else
  python -u playgroundManipulation_RSL_training.py \
      --env_name="$ENV_NAME" --impl="$IMPL" --seed="$TASKID" \
      --num_envs="$NUM_ENVS" --use_wandb=False \
      --use_rnn="$RNN_USE" --rnn_type="$RNN_TYPE" \
      --rnn_hidden_dim="$RNN_HIDDEN" --rnn_num_layers="$RNN_LAYERS" \
      --costate_coeff="$COSTATE" --history_len="$HISTORY_LEN" \
      --readout="$READOUT" \
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
