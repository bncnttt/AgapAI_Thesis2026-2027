"""Read-only checks of the installed runtime packages and cached models."""
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = "C:/AgapAI/.cache/matplotlib"
sys.path.insert(0, "C:/AgapAI/AgapAI/agapai/api")

import importlib

modules = [
    "atproto", "fastapi", "pymongo", "uvicorn", "calamancy", "psgc",
    "dateparser", "joblib", "sklearn", "nltk", "google.generativeai",
    "transformers", "torch", "sentencepiece", "langdetect", "pandas",
    "numpy", "matplotlib", "seaborn", "datasets", "accelerate", "openpyxl", "networkx",
]
for name in modules:
    importlib.import_module(name)
    print(f"Import OK: {name}", flush=True)

from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import snapshot_download
from nltk.corpus import wordnet
from agapai_api.ner_algorithm import get_nlp

for name in ("facebook/nllb-200-distilled-600M", "facebook/bart-large-mnli"):
    AutoConfig.from_pretrained(name, local_files_only=True)
    AutoTokenizer.from_pretrained(name, local_files_only=True)
    print(f"Cached model config/tokenizer OK: {name}", flush=True)

model_dir = Path("C:/AgapAI/AgapAI/agapai/api/agapai_api/models/agapai_multitask_xlmr/encoder")
AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
print("Saved classifier tokenizer OK", flush=True)

doc = get_nlp()("May baha sa Talisay, Cebu.")
assert any(ent.label_ == "LOC" for ent in doc.ents)
print("calamanCy NER model OK", flush=True)
assert wordnet.synsets("flood")
print("WordNet OK", flush=True)

from agapai_api.app import app
print(f"API import OK: {len(app.routes)} routes", flush=True)
