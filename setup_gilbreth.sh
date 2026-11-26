#!/usr/bin/env bash
# One-time setup on a Gilbreth login node (idempotent). Run it in a login shell so `module` exists:
#   cd /scratch/gilbreth/$USER/rlhf-ppo && bash -l setup_gilbreth.sh
# Override the defaults with environment variables:
#   SCRATCH   where the conda env lives       (default /scratch/gilbreth/$USER)
#   ENV       path of the conda env           (default $SCRATCH/envs/rlhf_ppo)
set -euo pipefail

PROJ="$(cd "$(dirname "$0")" && pwd)"
SCRATCH="${SCRATCH:-/scratch/gilbreth/$USER}"
ENV="${ENV:-$SCRATCH/envs/rlhf_ppo}"
export HF_HOME="$PROJ/hf_cache"                 # the notebook looks here by default
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"      # keep multi-GB wheels off the 25 GB home quota
mkdir -p "$HF_HOME" "$PROJ/out" "$PROJ/logs" "$PIP_CACHE_DIR"

module load conda 2>/dev/null || true

if [ ! -x "$ENV/bin/python" ]; then
  echo ">> creating conda env at $ENV"
  conda create -y -p "$ENV" python=3.11
fi

echo ">> installing python packages"
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install -r "$PROJ/requirements.txt"

echo ">> registering a Jupyter kernel (for Open OnDemand / JupyterLab)"
"$ENV/bin/python" -m ipykernel install --user --name rlhf_ppo --display-name "rlhf_ppo"

echo ">> pre-downloading IMDB, GPT-2 and the judge model into $HF_HOME"
"$ENV/bin/python" - <<'PY'
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2LMHeadModel, GPT2Model

load_dataset("stanfordnlp/imdb")
AutoTokenizer.from_pretrained("openai-community/gpt2")
GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
GPT2Model.from_pretrained("openai-community/gpt2")
judge = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
AutoTokenizer.from_pretrained(judge)
AutoModelForSequenceClassification.from_pretrained(judge)
print("downloads complete")
PY

echo ">> checking that everything loads without network access"
HF_HUB_OFFLINE=1 "$ENV/bin/python" - <<'PY'
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2LMHeadModel

print({k: len(v) for k, v in load_dataset("stanfordnlp/imdb").items()})
AutoTokenizer.from_pretrained("openai-community/gpt2")
GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print("offline load ok")
PY

echo ">> done. Either open rlhf_ppo_imdb.ipynb in JupyterLab on a GPU node with the 'rlhf_ppo' kernel and Run All,"
echo "   or submit it:  sbatch -A <account> -p a100-80gb -q standby run_notebook.sbatch"
