"""Fetch the NLP assets used by this checkout, reusing the user's model cache."""
import json
import shutil
from pathlib import Path

import nltk
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE


def prepare_model(repo_id, revision="main", patterns=None, ignore=None):
    print(f"Checking {repo_id} ({revision})...", flush=True)
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    cache = Path(HF_HUB_CACHE)
    snapshot = cache / ("models--" + repo_id.replace("/", "--")) / "snapshots" / info.sha
    from fnmatch import fnmatch
    required = [
        f for f in info.siblings
        if (not patterns or any(fnmatch(f.rfilename, p) for p in patterns))
        and not any(fnmatch(f.rfilename, p) for p in (ignore or []))
    ]
    missing_bytes = sum((f.size or 0) for f in required if not (snapshot / f.rfilename).is_file())
    free = shutil.disk_usage(cache).free
    print(f"Additional download: {missing_bytes / 2**20:.1f} MiB", flush=True)
    if missing_bytes + 200 * 2**20 > free:
        raise RuntimeError(f"Not enough disk space to fetch {repo_id}; need {missing_bytes / 2**30:.2f} GiB.")
    path = snapshot_download(repo_id, revision=revision, allow_patterns=patterns, ignore_patterns=ignore)
    print(f"Ready: {path}", flush=True)


prepare_model("ljvmiranda921/tl_calamancy_md", "0.2.0", ignore=["*.whl"])
prepare_model("facebook/nllb-200-distilled-600M", patterns=["*.json", "*.txt", "*.model", "*.safetensors"])
prepare_model("facebook/bart-large-mnli", patterns=["*.json", "*.txt", "*.safetensors"])
for name in ("wordnet", "omw-1.4"):
    if not nltk.download(name, raise_on_error=True):
        raise RuntimeError(f"Could not prepare NLTK {name}")
print("All NLP resources are available.", flush=True)
