# AgentPoison Colab Reproduction Runbook

Use this with `agentpoison_reproduction.ipynb`.

## 1. Start Colab

Open `agentpoison_reproduction.ipynb` in Colab, then choose:

```text
Runtime > Change runtime type > GPU
```

If Colab offers an H100/A100, use it. L4/T4 can still run the notebook, but trigger optimization will be slower.

## 2. Storage

Keep `USE_GOOGLE_DRIVE = True` in the first setup cell so outputs survive runtime disconnects:

```text
/content/drive/MyDrive/agentpoison_reproduction
```

If you set it to `False`, outputs go under `/content/agentpoison_reproduction` and disappear when Colab resets.

## 3. Clone the Repo

The notebook setup cell clones:

```bash
git clone https://github.com/BillChan226/AgentPoison.git
```

The local checkout in this project came from `https://github.com/AI-secure/AgentPoison.git`, which currently matches the official README content.

## 4. Install Dependencies

Colab already has a CUDA-compatible PyTorch build, so the notebook does not use the repo's old pinned conda environment. The dependency cell keeps `INSTALL_DEPS = True` and installs Colab/Python 3.12-compatible packages without pinning old `numpy`, `pandas`, `matplotlib`, or `torch` versions.

Run the compatibility patch cell immediately after cloning the repo. It fixes the newer `transformers` import failure for `RealmEmbedder` and a non-W&B `config` reference in `trigger_optimization.py`. The first reproduction uses DPR, so REALM-specific runs remain disabled on Colab unless you build an older Python/transformers environment.

If you need to skip installs in a later session, set:

```python
INSTALL_DEPS = False
```

Optional Agent-Driver packages are installed one at a time and are allowed to fail without blocking the `qa` and `ehr` reproduction.

## 5. Download Data

Download the unified AgentPoison datasets from the link in the official README. Put them here inside the cloned repo:

```text
AgentPoison/ReAct/database
AgentPoison/EhrAgent/database
```

Keep `AgentPoison/agentdriver/data` for the stretch goal only.

If the dataset is in Drive, either copy it into these folders or symlink it from Drive.

## 6. First Reproduction Pass

Run only one fast smoke pass before launching the full grid:

```bash
python algo/trigger_optimization.py \
  --agent qa \
  --algo ap \
  --model dpr-ctx_encoder-single-nq-base \
  --save_dir ../outputs/agentpoison_runs/triggers/qa/ap/dpr-ctx_encoder-single-nq-base/len10_seed0 \
  --num_iter 1000 \
  --num_grad_iter 30 \
  --per_gpu_eval_batch_size 64 \
  --num_cand 100 \
  --num_adv_passage_tokens 10 \
  --asr_threshold 0.5 \
  --ppl_filter \
  --plot
```

Do not add `--target_gradient_guidance` for the first Colab smoke run. In the released repo, that flag tries to load Llama-2 from an author-local cache path. Use the unguided DPR run first to verify the pipeline, then add a separate Llama/Hugging Face patch if you want the exact target-guidance ablation.

Then run attacked and benign StrategyQA:

```bash
python ReAct/run_strategyqa_gpt3.5.py --model dpr --task_type adv --save_dir ../outputs/agentpoison_runs/agent_runs/qa/dpr/adv/seed0
python ReAct/run_strategyqa_gpt3.5.py --model dpr --task_type benign --save_dir ../outputs/agentpoison_runs/agent_runs/qa/dpr/benign/seed0
```

Evaluate response JSONs:

```bash
python ReAct/eval.py -p PATH_TO_RESPONSE_JSON
```

## 7. Presentation Assets

The notebook saves slide assets under:

```text
outputs/figures
outputs/tables
outputs/presentation_asset_manifest.csv
```

Before making slides, rerun the manifest cell so every figure/table has a timestamped index.

## 8. Known Follow-Up

The current checkout does not expose `--seed` in `algo/trigger_optimization.py`. For strict three-seed reporting, add a small seed argument that sets `random`, `numpy`, and `torch` seeds before trigger optimization.
