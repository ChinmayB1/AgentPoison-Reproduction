#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/private/agentpoison_reproduction"
mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -d AgentPoison ]; then
  git clone https://github.com/BillChan226/AgentPoison.git
else
  echo "Repo exists: $ROOT/AgentPoison"
fi

python3 -m pip install --user -q \
  jsonlines shortuuid termcolor wandb openai peft timm webdataset \
  beautifulsoup4 gym python-Levenshtein wolframalpha replicate \
  omegaconf iopath sentence-transformers datasets seaborn scikit-learn

python3 - <<'PY'
from pathlib import Path

root = Path.home() / "private" / "agentpoison_reproduction" / "AgentPoison"

replacements = {
    root / "algo" / "trigger_optimization.py": {
        'root_dir = f"{args.save_dir}/{config.agent}/{config.algo}/{str(datetime.datetime.now())}"':
        'root_dir = f"{args.save_dir}/{args.agent}/{args.algo}/{str(datetime.datetime.now())}"',
    },
    root / "algo" / "utils.py": {
        """from transformers import (BertModel, 
                          BertTokenizer, 
                          AutoModelForCausalLM, 
                          LlamaForCausalLM, 
                          DPRContextEncoder,
                          AutoModel,
                          DPRQuestionEncoder,
                          RealmEmbedder,
                          RealmForOpenQA)""":
        """from transformers import (BertModel,
                          BertTokenizer,
                          AutoModelForCausalLM,
                          LlamaForCausalLM,
                          DPRContextEncoder,
                          AutoModel,
                          DPRQuestionEncoder)
try:
    from transformers import RealmEmbedder, RealmForOpenQA
except ImportError:
    class RealmEmbedder:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("RealmEmbedder unavailable; use dpr/ance/bge or older transformers.")

    class RealmForOpenQA:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("RealmForOpenQA unavailable; use dpr/ance/bge or older transformers.")""",
    },
    root / "ReAct" / "local_wikienv.py": {
        "from transformers import AutoTokenizer, DPRContextEncoder, RealmEmbedder":
        """from transformers import AutoTokenizer, DPRContextEncoder
try:
    from transformers import RealmEmbedder
except ImportError:
    class RealmEmbedder:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("RealmEmbedder unavailable; use dpr/ance/bge or older transformers.")""",
    },
    root / "agentdriver" / "memory" / "memory_agent.py": {
        "from transformers import (BertModel, BertTokenizer, DPRContextEncoder, AutoModel, RealmEmbedder, RealmForOpenQA)":
        """from transformers import BertModel, BertTokenizer, DPRContextEncoder, AutoModel
try:
    from transformers import RealmEmbedder, RealmForOpenQA
except ImportError:
    class RealmEmbedder:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("RealmEmbedder unavailable; use dpr/ance/bge or older transformers.")

    class RealmForOpenQA:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("RealmForOpenQA unavailable; use dpr/ance/bge or older transformers.")""",
    },
}

for path, mapping in replacements.items():
    text = path.read_text()
    changed = False
    for old, new in mapping.items():
        if old in text:
            text = text.replace(old, new)
            changed = True
    if changed:
        path.write_text(text)
        print("patched", path)
    else:
        print("already patched or pattern missing", path)
PY

cd "$ROOT/AgentPoison"
python3 - <<'PY'
import torch
from algo.utils import load_models
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
print("AgentPoison imports ok")
PY

du -sh . ReAct/database EhrAgent/database
