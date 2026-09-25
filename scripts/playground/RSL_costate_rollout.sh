#!/bin/bash -l
#SBATCH --job-name=fhn-costate
#SBATCH --partition=hawaii # lanai,molokai
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --time=01:00:00
#SBATCH -o logs/%x_%A_%a_output.log
#SBATCH -e logs/%x_%A_%a_error.log

# Roll trained policies from the initial conditions of the co-state dataset and
# log hidden-state trajectories for analysis.
#
# One run:
#   sbatch RSL_costate_rollout.sh --run=logs/FitzhughNagumo/rnn-gru-h64-l1_hist1_lr5e-4/job80950_seed1
#
# Every run under an architecture directory:
#   sbatch RSL_costate_rollout.sh --glob='logs/FitzhughNagumo/rnn-*/job*_seed*'
#

set -o pipefail

RUN=""
GLOB=""
DATA_DIR=""
ENV_NAME="FitzhughNagumo"
CHECKPOINT=""
DEVICE="cuda:0"
LOCAL="False"

usage() {
  cat <<'EOF'
Usage: sbatch RSL_costate_rollout.sh (--run=DIR | --glob=PATTERN) [options]

  --run=DIR          one training run directory
  --glob=PATTERN     quoted glob matching several run directories
  --data_dir=DIR     holds FHN_*_converged.npy  (default: env dir above --run)
  --env=NAME         custom env name                     (FitzhughNagumo)
  --checkpoint=FILE  specific model_*.pt         (default: highest numbered)
  --device=DEV                                            (cuda:0)
  --local            run in the current shell, no conda/slurm setup
EOF
}

for arg in "$@"; do
  case $arg in
    --run=*)        RUN="${arg#*=}" ;;
    --glob=*)       GLOB="${arg#*=}" ;;
    --data_dir=*)   DATA_DIR="${arg#*=}" ;;
    --env=*)        ENV_NAME="${arg#*=}" ;;
    --checkpoint=*) CHECKPOINT="${arg#*=}" ;;
    --device=*)     DEVICE="${arg#*=}" ;;
    --local)        LOCAL="True" ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "ERROR: unknown argument '$arg'"; usage; exit 1 ;;
  esac
done

if [ -z "$RUN" ] && [ -z "$GLOB" ]; then
  echo "ERROR: one of --run or --glob is required."
  usage
  exit 1
fi

if [ "$LOCAL" != "True" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate mj_playground2
  cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
fi

PROJECT_ROOT=$(realpath "${SLURM_SUBMIT_DIR:-.}/../..")
export PYTHONPATH=$PROJECT_ROOT

EXTRA=""
[ -n "$DATA_DIR" ]   && EXTRA="$EXTRA --data_dir=$DATA_DIR"
[ -n "$CHECKPOINT" ] && EXTRA="$EXTRA --checkpoint=$CHECKPOINT"

run_one() {
  local dir="$1"
  echo "=========================================================="
  echo " rollout: $dir"
  echo "=========================================================="
  python -u RSL_costate_rollout.py \
      --run_dir="$dir" --env_name="$ENV_NAME" --device="$DEVICE" $EXTRA
  local status=$?
  
  [ $status -ne 0 ] && echo "FAILED (exit $status): $dir"
  return $status
}

FAILED=0
if [ -n "$RUN" ]; then
  run_one "$RUN" || FAILED=1
else
  shopt -s nullglob
  MATCHES=($GLOB)
  if [ ${#MATCHES[@]} -eq 0 ]; then
    echo "ERROR: --glob matched nothing: $GLOB"
    exit 1
  fi
  echo "matched ${#MATCHES[@]} run directories"
  for dir in "${MATCHES[@]}"; do
    [ -d "$dir" ] || continue
    run_one "$dir" || FAILED=1
  done
fi

echo ""
if [ $FAILED -eq 0 ]; then
  echo "All rollouts completed. Results are in costate_rollout.npz per run dir."
else
  echo "One or more rollouts failed; see above."
fi
exit $FAILED