import time
from threading import Lock

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agapai_api.config import CLASSIFIER_MODEL_DIR

router = APIRouter(prefix="/classifier", tags=["classifier"])

_clf = None
_lock = Lock()
MAX_ITEMS = 500
CHUNK = 16


def trace(msg):
    print(f"[CLF ROUTE] {msg}", flush=True)


def get_classifier():
    global _clf
    with _lock:
        if _clf is None:
            trace(f"first request: loading model from {CLASSIFIER_MODEL_DIR}")
            started = time.time()
            from agapai_api.classify import Classifier
            _clf = Classifier(str(CLASSIFIER_MODEL_DIR))
            trace(f"model loaded in {time.time() - started:.1f}s")
    return _clf


class Item(BaseModel):
    id: str
    text: str


class PredictRequest(BaseModel):
    items: list[Item]


@router.post("/predict")
def predict(req: PredictRequest):
    trace(f"POST /classifier/predict with {len(req.items)} item(s)")
    if len(req.items) > MAX_ITEMS:
        raise HTTPException(400, f"Send at most {MAX_ITEMS} items per request.")
    try:
        clf = get_classifier()
    except Exception as e:
        trace(f"model load FAILED: {e!r}")
        raise HTTPException(500, f"Could not load classifier: {e!r}")

    rows = []
    started = time.time()
    for i in range(0, len(req.items), CHUNK):
        chunk = req.items[i:i + CHUNK]
        trace(f"chunk {i // CHUNK + 1}: items {i}-{i + len(chunk) - 1} ids={[c.id for c in chunk]}")
        preds = clf.predict([it.text or "" for it in chunk])
        for it, p in zip(chunk, preds):
            p = dict(p)
            p.pop("text", None)
            p["_id"] = it.id
            rows.append(p)
    trace(f"done: {len(rows)} row(s) in {time.time() - started:.1f}s")
    return {"rows": rows, "total": len(rows)}