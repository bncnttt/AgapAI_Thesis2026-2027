from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer



DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "agapai_multitask_xlmr"
MODEL_DIR = Path(os.getenv("AGAPAI_MODEL_DIR", DEFAULT_MODEL_DIR))
DEVICE = torch.device(os.getenv("AGAPAI_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
BATCH_SIZE = int(os.getenv("AGAPAI_BATCH_SIZE", "16"))

CLS_TASKS = ("role", "urgency", "disaster", "intent")


class MultiTaskXLMR(nn.Module):
    """Shared XLM-R encoder + one linear head per task. Must match the training notebook."""

    def __init__(self, encoder_dir: str | Path, head_sizes: dict[str, int], dropout: float = 0.1,
                 pooling: str = "cls_mean"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(str(encoder_dir))
        self.pooling = pooling
        hidden = self.encoder.config.hidden_size * (2 if pooling == "cls_mean" else 1)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, n) for task, n in head_sizes.items()})

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = h[:, 0]
        if self.pooling == "cls_mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = torch.cat([pooled, (h * m).sum(1) / m.sum(1).clamp(min=1)], dim=-1)
        pooled = self.dropout(pooled)
        return {task: head(pooled) for task, head in self.heads.items()}


class AgapaiClassifier:
    def __init__(self, model_dir: str | Path = MODEL_DIR, device: torch.device = DEVICE):
        model_dir = Path(model_dir)
        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"AGAPAI model not found at {model_dir}. Unzip agapai_multitask_xlmr_export.zip there "
                f"or set AGAPAI_MODEL_DIR.")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.device = device
        self.max_length = int(self.meta["max_length"])

        self.id2label = {t: {i: lbl for lbl, i in m.items()} for t, m in self.meta["label2id"].items()}
        self.categories: list[str] = self.meta["extraction_labels"]
        self.thresholds = torch.tensor([self.meta["thresholds"][c] for c in self.categories])
        intent_ids = self.meta["label2id"]["intent"]
        self.need_i, self.offer_i = intent_ids["need"], intent_ids["offer"]

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "encoder"))
        self.model = MultiTaskXLMR(model_dir / "encoder", self.meta["head_sizes"],
                                   self.meta.get("dropout", 0.1), self.meta.get("pooling", "cls"))
        self.model.heads.load_state_dict(torch.load(model_dir / "heads.pt", map_location="cpu"))
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = [t if isinstance(t, str) else "" for t in texts[start:start + BATCH_SIZE]]
            enc = self.tokenizer(batch, truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            probs = {t: torch.softmax(logits[t].float(), -1).cpu() for t in CLS_TASKS}
            res_p = torch.sigmoid(logits["extraction"].float()).cpu()

            for i, text in enumerate(batch):
                out: dict[str, Any] = {"text": text}
                for t in CLS_TASKS:
                    j = int(probs[t][i].argmax())
                    out[t] = self.id2label[t][j]
                    out[f"{t}_confidence"] = round(float(probs[t][i, j]), 4)

                found = {c: round(float(res_p[i, k]), 4) for k, c in enumerate(self.categories)
                         if res_p[i, k] > self.thresholds[k]}
                is_need = probs["intent"][i, self.need_i] >= probs["intent"][i, self.offer_i]
                out["needs"] = found if is_need else {}
                out["offers"] = {} if is_need else found
                results.append(out)
        return results


_classifier: AgapaiClassifier | None = None
_lock = threading.Lock()


def get_classifier() -> AgapaiClassifier:
    """Load the model once (thread-safe) and reuse it for every request."""
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:
                _classifier = AgapaiClassifier()
    return _classifier


def classify(text: str) -> dict[str, Any]:
    return get_classifier().predict([text])[0]


def classify_batch(texts: list[str]) -> list[dict[str, Any]]:
    return get_classifier().predict(list(texts))

# Disaster gate used by pipeline.py and retrieval.py (replaces disaster_relevance.py).


if __name__ == "__main__":
    import sys
    samples = sys.argv[1:] or [
        "Nanginahanglan mi ug sinina ug habol diri sa evacuation center sa Talisay",
        "Nasunog ang among balay sa Pasil, wala mi kapuy-an karon. Nanginahanglan mi ug sinina ug habol.",
        "We have relief packs with rice and bottled water ready in Lapu-Lapu. DM us.",
    ]
    for r in classify_batch(samples):
        print(json.dumps(r, ensure_ascii=False, indent=2))