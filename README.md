# Reproducing AgentPoison: Backdoor Attacks on RAG-Based LLM Agents
### UCSD DSC 291: Trustworthy Machine Learning (Spring 2026)
**Team Members:** Chinmay N, Chinmay B, Hayden, Suditi

https://medium.com/@cnadgir/your-rag-database-is-a-backdoor-replicating-and-defending-against-agentpoison-40589b6d7a10

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/)

This repository contains the reproduction, analysis, and defense extension of **AgentPoison** (Chen et al., NeurIPS 2024), a database-centric backdoor attack against RAG-based LLM agents. 

We replicate the attack across multiple LLM backbones (Qwen-2.5-7B, Llama-3-8B, and GPT-4o-mini) and propose a geometric defense to detect optimized query clustering in vector embedding spaces.

---

## 📂 Repository Structure

The repository is organized as follows:

```
├── EhrAgent/                  # Modified EHRAgent environment
├── ReAct/                     # Modified ReAct-StrategyQA environment
├── algo/                      # Core trigger optimization & baseline code
│   └── detector.py            # [NEW] E1 Geometric Defense implementation
├── notebooks/                 # [NEW] Interactive notebooks
│   └── agentpoison_reproduction.ipynb  # Main reproduction & evaluation workbook
├── report/                    # [NEW] Technical paper, blog, and media assets
│   ├── main.tex               # Complete LaTeX paper report (NeurIPS style)
│   ├── reproduction_report_blog.md  # Markdown draft of the Medium tutorial
│   ├── reproduction_report_blog.pdf  # Compiled PDF version of the blog tutorial
│   └── images/                # Generated figures and table images
├── scripts/
│   ├── cluster/               # [NEW] Cluster runbooks & parallel execution scripts
│   │   ├── dsmlp_*.sh         # Slurm/Kubernetes cluster launch shell scripts
│   │   ├── run_all_*.sh       # Evaluation automation pipelines
│   │   ├── run_experiments.py # Parameter sweep script for evaluations
│   │   └── sync_to_cluster.sh # Script to sync local files with DSMLP nodes
│   └── ...                    # Original baseline run scripts
├── environment.yml            # Conda environment specifications
└── README_original.md         # Original AgentPoison README documentation
```

---

## 🚀 Key Accomplishments and Contributions

1. **Replication & Verification**:
   - Replicated AgentPoison, CPA, and BadChain baselines across ReAct-StrategyQA and EHRAgent.
   - Evaluated commercial backbones (GPT-4o-mini) and open-weights models (Qwen-2.5-7B, Meta-Llama-3-8B).
   - Validated that AgentPoison achieves high end-to-end success (84.0% ASR-E2E on EHRAgent with GPT-4o-mini).
2. **Fault-Tolerant Optimization**:
   - Implemented a checkpointing layer directly into `trigger_optimization.py` to save running state and prevent progress loss from shared cluster pod evictions (350+ GPU hours total).
3. **The "Noisy" Backdoor Paradox**:
   - Documented a semantic bias bug: retrievers naturally fetch poisoned documents on benign queries related to targeted topics, leading to a localized denial-of-service (benign accuracy drops to 60.0% even without trigger $T$).
4. **Geometric Defense (E1)**:
   - Designed a `GeometricDetector` inside `algo/detector.py` that monitors retrieved embedding similarity and vocabulary repetitiveness, flagging AgentPoison with a **98.0% True Positive Rate**.
5. **Adaptive Attacker Sweep**:
   - Built and evaluated an adaptive attacker that penalizes embedding variance to evade detection, mapping the trade-off curve between dispersion and retrieval rate.

---

## 🛠️ Installation and Setup

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/chinmaynadgir/AgentPoison-Reproduction.git
cd AgentPoison-Reproduction
conda env create -f environment.yml
conda activate agentpoison
```

### 2. Configure API Keys
To run evaluations against API-based LLMs, set up your OpenAI API key in your environment variables:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

---

## 💻 How to Run

### Interactive Workbook
The easiest way to inspect intermediate outputs, run evaluations, and reproduce the paper's figures is through the Jupyter Notebook:
- Located at [notebooks/agentpoison_reproduction.ipynb](notebooks/agentpoison_reproduction.ipynb).
- It provides a step-by-step walk-through of RAG setups, database poisoning, evaluation traces, and plotting functions.

### Trigger Optimization
To run trigger optimization for ReAct-StrategyQA with a DPR embedder:
```bash
python algo/trigger_optimization.py \
    --agent qa \
    --algo ap \
    --model dpr-ctx_encoder-single-nq-base \
    --save_dir ./results \
    --ppl_filter \
    --target_gradient_guidance \
    --num_adv_passage_tokens 10
```

### Running Cluster Experiments
To run parameter sweeps or parallel workloads on an academic GPU cluster (e.g., Slurm/Kubernetes):
```bash
cd scripts/cluster
# Sync files to cluster nodes
./sync_to_cluster.sh
# Execute StrategyQA trigger search
./dsmlp_run_qa_trigger_bs8_iter25.sh
```

---

## 📊 Summary of Main Results

Detailed numbers comparing paper reported rates against our replication (GPT-4o-mini):

| Agent | Method | ASR-r (Paper) | ASR-r (Ours) | ASR-t (Paper) | ASR-t (Ours) | ACC (Paper) | ACC (Ours) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ReAct-SQA** | AgentPoison | 81.3% | 92.0% | 60.5% | 52.0% | 66.7% | 60.0% |
| **EHRAgent**   | AgentPoison | 80.0% | 92.0% | 61.0% | 84.0% | 70.0% | 100.0% |
| **ReAct-SQA** | CPA         | 17.3% | 96.0% | 14.2% | 56.0% | 67.1% | 60.0% |
| **EHRAgent**   | CPA         | 24.0% | 92.0% | 17.0% | 68.0% | 70.0% | 100.0% |
| **ReAct-SQA** | BadChain    | 5.2%  | 40.0% | 10.1% | 48.0% | 66.9% | 60.0% |
| **EHRAgent**   | BadChain    | 10.0% | 84.0% | 8.0%  | 88.0% | 70.0% | 100.0% |

All figures and plots generated during the reproduction can be found under the [report/images/](report/images/) directory.

---

## 📄 License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
