import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agapai_api.classify import banner, end_run_log, get_classifier, start_run_log
from agapai_api.classify import trace as clf_trace

router = APIRouter(prefix="/classifier", tags=["classifier"])

MAX_ITEMS = 500
CHUNK = 16


def trace(msg):
    clf_trace("ROUTE", msg)


class Item(BaseModel):
    id: str
    text: str


class PredictRequest(BaseModel):
    items: list[Item]


@router.post("/predict")
def predict(req: PredictRequest):
    start_run_log("run")          # new file: logs/classifier/classifier_<timestamp>.log
    try:
        banner("CONNECTED TO THE SYSTEM")
        trace(f"POST /classifier/predict with {len(req.items)} item(s)")
        if len(req.items) > MAX_ITEMS:
            raise HTTPException(400, f"Send at most {MAX_ITEMS} items per request.")
        try:
            clf = get_classifier()    # the one shared model (same one routes.py loads at startup)
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
    finally:
        end_run_log()
