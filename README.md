# RLHF with PPO

A full RLHF pipeline on GPT-2 (124M) in one notebook, `rlhf_ppo_imdb.ipynb`. It fine-tunes GPT-2 on IMDB reviews, trains a reward model on preference pairs with the Bradley-Terry loss, then uses PPO with a KL penalty to steer the model toward positive reviews.

On 2,000 held-out prompts, an independent sentiment classifier rated 99.2% of PPO outputs positive, up from 47.9% before PPO. The full run takes about 33 minutes on one A100 80GB. Details are in `RLHF-REPORT.md`.

## Run

```bash
pip install -r requirements.txt
jupyter lab rlhf_ppo_imdb.ipynb
```

`setup_gilbreth.sh` and `run_notebook.sbatch` run it on Purdue's Gilbreth cluster. Without a GPU, the notebook runs a short smoke test instead.
