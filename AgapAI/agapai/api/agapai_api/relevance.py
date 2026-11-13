"""Semantic retrieval gate; reuse the application's local NLLB and BART models."""
from math import isfinite

DISASTER_LABELS = (
    "physical flooding with water",
    "a destructive fire burning buildings",
    "an earthquake, typhoon, or landslide",
)
NON_DISASTER_LABELS = (
    "money or wealth described figuratively as a flood",
    "shopping, entertainment, or other non-disaster topics",
)
# Require the winning interpretation to be a physical disaster, not merely
# a high score against a single broad negative label. Still needs validation
# on a representative labeled dataset; these are not calibrated probabilities.
RELEVANCE_THRESHOLD = 0.60


def analyze_disaster_relevance(text):
    from agapai_api.local_models import (
        clean_for_classification, detect_source_language_code,
        get_zero_shot_classifier, translate_text,
    )

    cleaned = clean_for_classification(text)
    if not cleaned:
        return {"relevant": False, "confidence": 0.0, "label": NON_DISASTER_LABELS[-1]}
    source = detect_source_language_code(cleaned)
    english = translate_text(cleaned, source, "eng_Latn")
    if not english or not english.strip():
        raise ValueError("Empty relevance translation")
    result = get_zero_shot_classifier()(
        english, candidate_labels=list(DISASTER_LABELS + NON_DISASTER_LABELS),
        hypothesis_template="This text is about {}.", multi_label=False,
    )
    scores = dict(zip(result["labels"], result["scores"]))
    scores = {label: float(scores[label]) for label in DISASTER_LABELS + NON_DISASTER_LABELS}
    if not all(isfinite(score) and 0 <= score <= 1 for score in scores.values()):
        raise ValueError("Invalid relevance scores")
    label = max(scores, key=scores.get)
    confidence = scores[label]
    return {
        "relevant": label in DISASTER_LABELS and confidence >= RELEVANCE_THRESHOLD,
        "confidence": confidence,
        "label": label,
        "scores": scores,
        "source_language": source,
        "translated_text": english,
        "threshold": RELEVANCE_THRESHOLD,
    }
