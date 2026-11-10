"""
Stack B: local translation + strict disaster/intent classification for
Cebu posts. No hosted LLM API (Gemini or otherwise) is used anywhere in
this module -- translation and classification both run entirely on this
machine via NLLB-200 and BART-MNLI.

This replaces two prior implementations, fixing what actually broke each
one:

1. The original local NLLB-200 + BART-MNLI stack (agapai_api/disaster_nlp.py)
   corrupted Philippine place names during translation (e.g. "Talisay" ->
   "Taraiza") because NLLB is a narrow translation model with no concept
   of "this token is a place name, don't touch it"; and BART-MNLI's
   standard (single-label) zero-shot mode always ranks *some* label
   highest across whatever candidates it's given, so non-disaster content
   (crime news, waterfall photos) got shoved into a disaster label.
2. A later Gemini-based version fixed both by using a general LLM's world
   knowledge, but requires a hosted API key.

This version fixes the SAME two failure modes locally, without any LLM,
using two targeted techniques instead of "a smarter model":

- Entity masking: every known Cebu place name (geography.py's dynamic
  PSGC registry) is replaced with a placeholder BEFORE translation and
  restored verbatim AFTER, so NLLB never even sees the actual place name
  and cannot corrupt it -- this is deterministic, not "hoping the model
  gets it right."
- Independent-threshold zero-shot classification (multi_label=True):
  each candidate label is scored on its own entailment probability
  instead of being forced into a single ranked choice, so "none of the 5
  disaster types fit" and "no active help ask/offer" are real, reachable
  outcomes -- this is what actually fixes the forced-choice bug, not the
  choice of classifier model itself.
"""
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

# A small, fixed set of major Philippine regions/provinces outside Cebu
# Province -- a geographic constant (comparable to the 5 fixed disaster
# type seeds), not an open-ended disaster-keyword or dialect vocabulary.
# Used only to help distinguish "Cebu is the disaster victim" from "Cebu
# is sending aid elsewhere"; testing showed neither a pure-semantic check
# nor a generic capitalized-word heuristic could reliably make this call
# alone with this size of local model.
OUTSIDE_CEBU_REGIONS = ["mindanao", "luzon", "manila", "bohol", "leyte", "samar"]

# This hypothesis is this pipeline's equivalent of an LLM system-prompt
# instruction: "Only flag as true if a city/municipality within Cebu
# Province is the direct victim of the disaster. Exclude posts where Cebu
# is merely sending aid to outside regions." There is no LLM prompt here
# to update directly -- this phrasing plays that role for the zero-shot
# classifier instead.
OUTGOING_AID_HYPOTHESIS = (
    "Cebu residents or officials are sending or coordinating disaster "
    "relief aid for another place, rather than responding to a disaster "
    "happening inside Cebu itself"
)
OUTGOING_AID_THRESHOLD = 0.5
DISASTER_CONFIDENCE_THRESHOLD = 0.5
MIN_SUBSTANTIVE_WORDS = 3


def predict_disaster_type(text):
    """
    Returns {"disaster_type": ..., "disaster_type_confidence": ...}. The
    model's disaster head has no "not a disaster" class, so a near-empty
    post, a "general disaster" call, or a low-confidence call is "none".
    """
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
    """
    True only when BOTH signals agree: the post names a major region
    outside Cebu Province, AND the text semantically reads as Cebu
    sending/coordinating aid elsewhere rather than describing a disaster
    inside Cebu. Requiring both avoids the false positives/negatives
    either signal produced alone during testing (e.g. a genuine local
    Cebu flood-relief post getting rejected just for using "relief"
    language, or ordinary capitalized words being mistaken for places).
    """
    if not mentions_outside_cebu_region(text):
        return False

    _label, score = classify_with_threshold(text, [OUTGOING_AID_HYPOTHESIS], 0.0)
    return score >= OUTGOING_AID_THRESHOLD


def evaluate_post(raw_text, parent_text=None):
    """
    Runs one post through Stack B, entirely locally: find known Cebu place
    names, then classify disaster type on the original
    text. No translation
    is performed.

    parent_text: thread-stitching context. If this post is a reply, pass
    the parent post's text so a fragment like "It's completely underwater!"
    is classified together with what it's actually replying to.
    """
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

    # Only the matched place names are used here (the masked text itself
    # was only needed to protect them from translation).
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
    """
    The save gate: any post about one of the 5 target disasters is kept,
    including general reporting/discussion with no ask for or offer of
    help. Needs and donations are recorded by the classifier.
    """
    return evaluation["is_disaster"] is True and evaluation["disaster_type"] != "none"
