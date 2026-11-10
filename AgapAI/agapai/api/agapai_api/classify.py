"""classify.py - standalone inference for the AgapAI multi-task XLM-R model.

Expects the folder produced by Step 12 of the notebook:
    export/
      encoder/     (HF encoder + tokenizer)
      heads.pt
      meta.json

Usage:
    from classify import Classifier
    clf = Classifier("path/to/export")
    clf.predict("Need food and water in Brgy. X, flooded since last night")
"""
import json
import os
import re

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

try:
    import ftfy
except ImportError:                      # optional, but matches training preprocessing
    ftfy = None


# ----------------------------------------------------------------------------
# Tracing: set AGAPAI_TRACE=0 in the environment to silence all trace output.
# ----------------------------------------------------------------------------
TRACE = os.getenv("AGAPAI_TRACE", "1") != "0"


def trace(tag, msg=""):
    if TRACE:
        print(f"[CLF {tag}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Text cleaning: must match Step 2 of the notebook (excel escapes -> ftfy -> strip)
# ----------------------------------------------------------------------------
_XESC = re.compile(r"_x([0-9A-Fa-f]{4})_")


def decode_excel_escapes(text):
    return _XESC.sub(lambda m: chr(int(m.group(1), 16)), text)


def clean_text(text):
    text = decode_excel_escapes(str(text))
    if ftfy is not None:
        text = ftfy.fix_text(text)
    return text.strip()


# ----------------------------------------------------------------------------
# Model: copied verbatim from Step 7 (names must match for load_state_dict)
# ----------------------------------------------------------------------------
class MultiTaskXLMR(nn.Module):
    def __init__(self, model_name, head_sizes, dropout=0.1, pooling="cls_mean"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden_dim = self.encoder.config.hidden_size
        pool_dim = hidden_dim * (2 if pooling == "cls_mean" else 1)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {task: nn.Linear(pool_dim, n) for task, n in head_sizes.items()})

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        cls_vec = h[:, 0]
        if self.pooling == "cls_mean":
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            mean_vec = (h * m).sum(1) / m.sum(1).clamp(min=1)
            pooled = torch.cat([cls_vec, mean_vec], dim=-1)
        else:
            pooled = cls_vec
        pooled = self.dropout(pooled)
        return {task: head(pooled) for task, head in self.heads.items()}


# ----------------------------------------------------------------------------
# Resource selection: copied from Step 8
# ----------------------------------------------------------------------------
def select_topk_resources(resource_logits, count_output):
    scores = torch.as_tensor(resource_logits)
    counts = torch.as_tensor(count_output, device=scores.device).reshape(-1)
    counts = torch.round(counts).long().clamp(0, scores.shape[1])
    pred = torch.zeros_like(scores, dtype=torch.int32)
    for row_idx, k in enumerate(counts.tolist()):
        if k > 0:
            top_idx = torch.topk(scores[row_idx], k=int(k), largest=True).indices
            pred[row_idx, top_idx] = 1
    return pred


# ----------------------------------------------------------------------------
# Classifier: load_pipeline (Step 12) + predict (Step 11), without notebook globals
# ----------------------------------------------------------------------------
class Classifier:
    def __init__(self, export_dir, device=None):
        trace("LOAD 1/5", f"export_dir = {export_dir}")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        trace("LOAD 1/5", f"device = {self.device}")

        with open(os.path.join(export_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        trace("LOAD 2/5", f"meta.json read. keys = {list(self.meta)}")

        self.label2id = self.meta["label2id"]
        self.id2label = {t: {i: l for l, i in m.items()}
                         for t, m in self.label2id.items()}
        self.extraction_labels = self.meta["extraction_labels"]
        self.max_length = self.meta["max_length"]
        self.need_id = self.label2id["intent"]["need"]
        self.offer_id = self.label2id["intent"]["offer"]
        trace("LOAD 3/5", f"classification tasks = {list(self.label2id)}")
        for t, m in self.label2id.items():
            trace("LOAD 3/5", f"  {t}: {list(m)}")
        trace("LOAD 3/5", f"extraction labels ({len(self.extraction_labels)}) = {self.extraction_labels}")
        trace("LOAD 3/5", f"max_length = {self.max_length}, need_id = {self.need_id}, offer_id = {self.offer_id}")

        enc_dir = os.path.join(export_dir, "encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(enc_dir)
        trace("LOAD 4/5", f"tokenizer loaded from {enc_dir}")
        self.model = MultiTaskXLMR(enc_dir, self.meta["head_sizes"],
                                   self.meta["dropout"], self.meta["pooling"])
        self.model.heads.load_state_dict(
            torch.load(os.path.join(export_dir, "heads.pt"), map_location="cpu"))
        self.model = self.model.float().to(self.device).eval()   # fp16 export -> fp32
        trace("LOAD 5/5", f"encoder + heads loaded. head sizes = {self.meta['head_sizes']}")
        trace("LOAD 5/5", "model ready (eval mode, fp32)")

    @torch.no_grad()
    def predict(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        trace("STEP 1", f"received {len(texts)} text(s)")
        for i, t in enumerate(texts):
            trace("STEP 1", f"  [{i}] {t!r}")

        cleaned = [clean_text(t) for t in texts]
        trace("STEP 2", f"cleaning (excel escapes -> ftfy={'on' if ftfy else 'OFF, not installed'} -> strip)")
        for i, (raw, c) in enumerate(zip(texts, cleaned)):
            trace("STEP 2", f"  [{i}] changed = {raw != c}")
            trace("STEP 2", f"  [{i}] cleaned = {c!r}")

        enc = self.tokenizer(cleaned, truncation=True, max_length=self.max_length,
                             padding=True, return_tensors="pt").to(self.device)
        trace("STEP 3", f"tokenized. input_ids shape = {tuple(enc['input_ids'].shape)} (batch, padded_len)")
        for i in range(len(cleaned)):
            keep = enc["attention_mask"][i].bool()
            ids = enc["input_ids"][i][keep].tolist()
            toks = self.tokenizer.convert_ids_to_tokens(ids)
            full_len = len(self.tokenizer(cleaned[i])["input_ids"])
            trace("STEP 3", f"  [{i}] {len(ids)} tokens kept (full length {full_len}, "
                            f"max {self.max_length}, truncated = {full_len > self.max_length})")
            trace("STEP 3", f"  [{i}] tokens = {toks}")

        logits = self.model(enc["input_ids"], enc["attention_mask"])
        trace("STEP 4", "forward pass done (encoder -> cls+mean pooling -> heads). logits shapes:")
        for name, tensor in logits.items():
            trace("STEP 4", f"  {name}: {tuple(tensor.shape)}")

        cls_tasks = list(self.label2id)
        probs = {t: torch.softmax(logits[t], -1).cpu() for t in cls_tasks}
        ext_p = torch.sigmoid(logits["extraction"]).cpu()
        resource_sel = select_topk_resources(logits["extraction"].cpu(),
                                             logits["resource_count"].cpu())

        for i in range(len(texts)):
            trace("STEP 5", f"[{i}] softmax per classification task (sorted, best first):")
            for t in cls_tasks:
                ranked = sorted(
                    ((self.id2label[t][k], float(probs[t][i, k])) for k in range(probs[t].shape[1])),
                    key=lambda x: x[1], reverse=True)
                trace("STEP 5", f"  {t:<10} -> " + " | ".join(f"{n} {p:.3f}" for n, p in ranked))

            raw_count = float(logits["resource_count"].cpu().reshape(-1)[i])
            k_sel = int(resource_sel[i].sum())
            trace("STEP 6", f"[{i}] resource_count regression output = {raw_count:.3f} "
                            f"-> rounded/clamped k = {k_sel}")
            top = sorted(range(len(self.extraction_labels)),
                         key=lambda k: float(ext_p[i, k]), reverse=True)[:10]
            trace("STEP 6", f"[{i}] top-10 extraction sigmoid probs ('*' = selected by top-k):")
            for k in top:
                mark = "*" if int(resource_sel[i, k]) else " "
                trace("STEP 6", f"  {mark} {self.extraction_labels[k]:<28} {float(ext_p[i, k]):.3f}")

        out = []
        for i, text in enumerate(texts):
            r = {"text": text}
            for t in cls_tasks:
                j = int(probs[t][i].argmax())
                r[t] = self.id2label[t][j]
                r[f"{t}_conf"] = float(probs[t][i, j])

            intent_id = int(probs["intent"][i].argmax())
            idx = torch.where(resource_sel[i].bool())[0].tolist()
            trace("STEP 7", f"[{i}] intent = {r['intent']!r} (id {intent_id}); "
                            f"selected resource ids = {idx}")
            if intent_id not in (self.need_id, self.offer_id):
                trace("STEP 7", f"[{i}] intent is neither need nor offer -> resources cleared")
                idx = []
            idx = sorted(idx, key=lambda k: float(ext_p[i, k]), reverse=True)
            found = {self.extraction_labels[k]: round(float(ext_p[i, k]), 3)
                     for k in idx}

            r["resource_count"] = len(found)
            r["needs"] = found if intent_id == self.need_id else {}
            r["offers"] = found if intent_id == self.offer_id else {}
            trace("STEP 8", f"[{i}] routed to: "
                            f"{'needs' if intent_id == self.need_id else 'offers' if intent_id == self.offer_id else 'neither'}"
                            f" -> {found}")
            trace("STEP 9", f"[{i}] FINAL = {json.dumps({k: v for k, v in r.items() if k != 'text'}, ensure_ascii=False)}")
            out.append(r)
        return out[0] if single else out


# ----------------------------------------------------------------------------
# Compatibility layer: keeps `from agapai_api.classify import classify, get_classifier`
# working for pipeline.py / routes.py. One shared model instance, loaded lazily.
# ----------------------------------------------------------------------------
_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        from agapai_api.config import CLASSIFIER_MODEL_DIR   # lazy: keeps `python classify.py` usable
        trace("API", f"get_classifier(): first call, loading model from {CLASSIFIER_MODEL_DIR}")
        _classifier = Classifier(str(CLASSIFIER_MODEL_DIR))
    return _classifier


def classify(text):
    """Single text -> dict. Adds `<task>_confidence` aliases (e.g. disaster_confidence)
    next to the `<task>_conf` keys, because pipeline.py reads `disaster_confidence`."""
    trace("API", f"classify() called with {str(text)[:80]!r}")
    result = dict(get_classifier().predict(text))
    for key in list(result):
        if key.endswith("_conf"):
            result[key + "idence"] = result[key]
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        sys.exit("usage: python -m agapai_api.classify <export_dir> \"<post text>\"")
    clf = Classifier(sys.argv[1])
    print(json.dumps(clf.predict(sys.argv[2]), indent=2, ensure_ascii=False))