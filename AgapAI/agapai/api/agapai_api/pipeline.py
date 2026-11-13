
import re

from agapai_api.classify import classify
from agapai_api.geography import strip_dateline
from agapai_api.local_models import (
    classify_with_threshold,
    clean_for_classification,
    is_confidently_foreign_language,
    mask_cebu_locations,
)

DISASTER_TYPES = ["flood", "earthquake", "typhoon", "fire", "landslide"]

OUTSIDE_CEBU_REGIONS = ["mindanao", "luzon", "manila", "bohol", "leyte", "samar"]

OUTGOING_AID_HYPOTHESIS = (
    "Cebu residents or officials are sending or coordinating disaster "
    "relief aid for another place, rather than responding to a disaster "
    "happening inside Cebu itself"
)
OUTGOING_AID_THRESHOLD = 0.5
DISASTER_CONFIDENCE_THRESHOLD = 0.5
MIN_SUBSTANTIVE_WORDS = 3


def predict_disaster_type(text):
    cleaned_text = clean_for_classification(text or "")
    if len(cleaned_text.split()) < MIN_SUBSTANTIVE_WORDS:
        return {"disaster_type": "none", "disaster_type_confidence": 0.0}

    prediction = classify(cleaned_text)
    disaster_type = prediction["disaster"]
    confidence = prediction["disaster_confidence"]
    if disaster_type not in DISASTER_TYPES or confidence < DISASTER_CONFIDENCE_THRESHOLD:
        return {"disaster_type": "none", "disaster_type_confidence": confidence}
    return {"disaster_type": disaster_type, "disaster_type_confidence": confidence}

def mentions_outside_cebu_region(text):
    lowered = text.lower()
    return any(
        re.search(r'\b' + re.escape(region) + r'\b', lowered)
        for region in OUTSIDE_CEBU_REGIONS
    )


def is_outgoing_aid_post(text):
    if not mentions_outside_cebu_region(text):
        return False

    _label, score = classify_with_threshold(text, [OUTGOING_AID_HYPOTHESIS], 0.0)
    return score >= OUTGOING_AID_THRESHOLD


def evaluate_post(raw_text, parent_text=None):
    if not raw_text or not raw_text.strip():
        return {
            "is_disaster": False,
            "disaster_type": "none",
            "disaster_type_confidence": 0.0,
            "extracted_locations": [],
        }

    raw_text = strip_dateline(raw_text)

    if is_confidently_foreign_language(raw_text):
        print(
            "[AGAPAI LOG] Rejected: confidently detected as a language "
            "unrelated to the Philippines -- out of scope."
        )
        return {
            "is_disaster": False,
            "disaster_type": "none",
            "disaster_type_confidence": 0.0,
            "extracted_locations": [],
        }

    _masked_text, placeholder_map = mask_cebu_locations(raw_text)

    extracted_locations = sorted(set(placeholder_map.values()))

    classification_input = raw_text

    has_parent_context = bool(parent_text and parent_text.strip())
    if has_parent_context:
        classification_input = f"{parent_text} {classification_input}"

    disaster_prediction = predict_disaster_type(classification_input)
    disaster_type = disaster_prediction["disaster_type"]
    is_disaster = disaster_type != "none"

    if is_disaster and is_outgoing_aid_post(classification_input):
        print(
            "[AGAPAI LOG] Discarded as outgoing aid: Cebu is sending relief "
            "to an outside region, not the disaster victim."
        )
        is_disaster = False
        disaster_type = None

    return {
        "is_disaster": is_disaster,
        "disaster_type": disaster_type if is_disaster else "none",
        "disaster_type_confidence": disaster_prediction["disaster_type_confidence"],
        "extracted_locations": extracted_locations,
    }


def is_actionable(evaluation):
    return evaluation["is_disaster"] is True and evaluation["disaster_type"] != "none"
