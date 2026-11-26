#!/bin/bash
# Run the whole pipeline headless on one A100. Submit from the project directory:
#   sbatch -A <account> -p a100-80gb -q standby run_notebook.sbatch
# The notebook is executed in place, so it comes back with all outputs filled in.
# Settings can be changed without editing the notebook, for example:
#   sbatch ... --export=ALL,RLHF_CFG='{"ppo_kl_target": 8}' run_notebook.sbatch
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -J rlhf-ppo
#SBATCH -o logs/%x-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV="${ENV:-/scratch/gilbreth/$USER/envs/rlhf_ppo}"
TARGET="${TARGET:-rlhf_ppo_imdb.ipynb}"
KERNEL="${KERNEL:-rlhf_ppo}"                              # the kernel registered by setup_gilbreth.sh
export HF_HOME="$PWD/hf_cache" HF_HUB_OFFLINE=1          # models and data were downloaded by setup_gilbreth.sh
export TOKENIZERS_PARALLELISM=true OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1

hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [[ "$TARGET" == *.py ]]; then
  "$ENV/bin/python" "$TARGET"
else
  "$ENV/bin/jupyter" nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name="$KERNEL" "$TARGET"
fi
