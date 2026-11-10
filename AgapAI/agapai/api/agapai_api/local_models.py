"""
Shared local (non-API, non-LLM) NLP models for AgapAI: NLLB-200 for
translation and BART-MNLI for zero-shot classification, both run
entirely on-device. No Gemini or other hosted LLM is used anywhere in
this module. Centralized here so corpus.py and pipeline.py share one
cached model instance instead of each loading these multi-GB models
separately.
"""
import re
from threading import RLock

import langdetect
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from agapai_api.geography import PSGC_CEBU_ALL_LOCATIONS

TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"
ZERO_SHOT_MODEL_NAME = "facebook/bart-large-mnli"

_device = 0 if torch.cuda.is_available() else -1
_translation_tokenizer = None
_translation_model = None
_zero_shot_classifier = None
_translation_lock = RLock()


def get_translation_model():
    global _translation_tokenizer, _translation_model
    if _translation_model is None:
        print("[AGAPAI LOG] Loading local NLLB-200 translation model (first call only)...")
        _translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        _translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
    return _translation_tokenizer, _translation_model


def get_zero_shot_classifier():
    global _zero_shot_classifier
    if _zero_shot_classifier is None:
        print("[AGAPAI LOG] Loading local zero-shot classification model (first call only)...")
        _zero_shot_classifier = pipeline(
            "zero-shot-classification", model=ZERO_SHOT_MODEL_NAME, device=_device
        )
    return _zero_shot_classifier


def detect_source_language_code(text):
    """
    Guesses an NLLB-200 source-language code at runtime. langdetect has no
    distinct Cebuano class, so anything not confidently English or Tagalog
    defaults to Cebuano -- the dominant native language in Cebu Province,
    this project's area of interest. Callers should check
    is_confidently_foreign_language() first: this function alone would
    otherwise happily mistranslate e.g. Portuguese text as if it were
    Cebuano, producing garbage that can spuriously pass classification.
    """
    try:
        detected = langdetect.detect(text)
    except Exception:
        detected = None

    if detected == "en":
        return "eng_Latn"
    if detected == "tl":
        return "tgl_Latn"
    return "ceb_Latn"


# A small, fixed set of ISO 639-1 codes for languages clearly unrelated to
# the Philippines -- language-family metadata, not a disaster keyword or
# place-name list (comparable to the fixed OUTSIDE_CEBU_REGIONS constant
# used elsewhere in this pipeline). This is a targeted BLOCKLIST, not a
# strict "only en/tl allowed" allowlist: langdetect has no distinct
# Cebuano class, so genuine Cebuano text often gets misdetected as some
# unrelated language, and an allowlist would reject it outright. Only
# reject when langdetect is highly confident AND the detected language is
# one of these specific, well-detected, obviously-irrelevant ones.
CLEARLY_FOREIGN_LANGUAGE_CODES = {
    "pt", "es", "fr", "de", "it", "nl", "ru", "ja", "ko", "zh-cn", "zh-tw",
    "ar", "th", "vi", "tr", "pl", "sv", "el", "id", "ms",
}
FOREIGN_LANGUAGE_CONFIDENCE_THRESHOLD = 0.90


def is_confidently_foreign_language(text):
    """
    True only when langdetect is highly confident the text is written in
    a language clearly unrelated to the Philippines. Used as an early
    reject gate before translation, so e.g. a confidently-Portuguese post
    never gets mistranslated as Cebuano and fed to the classifier.
    """
    try:
        detected = langdetect.detect_langs(text)
    except Exception:
        return False

    if not detected:
        return False

    top = detected[0]
    return top.lang in CLEARLY_FOREIGN_LANGUAGE_CODES and top.prob >= FOREIGN_LANGUAGE_CONFIDENCE_THRESHOLD


def translate_text(text, src_lang, tgt_lang="eng_Latn", max_length=256):
    if not text or not text.strip():
        return text

    if src_lang == tgt_lang:
        return text

    # The cached tokenizer's source language is mutable across API requests.
    with _translation_lock:
        tokenizer, model = get_translation_model()
        tokenizer.src_lang = src_lang
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded, forced_bos_token_id=forced_bos_token_id, max_length=max_length
            )
        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated[0] if translated else text


def mask_cebu_locations(text):
    """
    Replaces every known PSGC Cebu location mentioned in text with a
    placeholder token before translation, so the NMT model never sees --
    and therefore cannot hallucinate or mangle -- the actual place name
    (the "Talisay" -> "Taraiza" failure mode). Returns
    (masked_text, {placeholder: original_matched_text}).
    """
    lowered = text.lower()
    found_locations = sorted(
        (
            location for location in PSGC_CEBU_ALL_LOCATIONS
            if len(location) >= 4 and re.search(r"\b" + re.escape(location) + r"\b", lowered)
        ),
        key=len,
        reverse=True,
    )

    masked_text = text
    placeholder_map = {}
    for index, location in enumerate(found_locations):
        pattern = r"\b" + re.escape(location) + r"\b"
        match = re.search(pattern, masked_text, flags=re.IGNORECASE)
        if not match:
            continue
        placeholder = f"XPLACEHOLDERX{index}X"
        placeholder_map[placeholder] = match.group(0)
        masked_text = re.sub(pattern, placeholder, masked_text, flags=re.IGNORECASE)

    return masked_text, placeholder_map


def restore_locations(translated_text, placeholder_map):
    restored = translated_text
    for placeholder, original_matched_text in placeholder_map.items():
        restored = re.sub(re.escape(placeholder), original_matched_text, restored, flags=re.IGNORECASE)
    return restored


def clean_for_classification(text):
    """
    Strips social-media formatting noise before zero-shot classification.
    BART-MNLI is noticeably less robust than a large LLM to superficial
    formatting: a literal "#" character attached to an otherwise-benign
    word (e.g. "#waterfall") was found to swing its flood-entailment score
    from ~0.005 to ~0.65, a false positive purely from the punctuation --
    stripping the "#" while keeping the word underneath fixes it.

    Also strips bare (schemeless) links, e.g. "blazenwingscorner.space/
    TuboOpensCebu" -- Bluesky's own link-facet rendering commonly omits
    "http(s)://" and "www.", so the plain "https?://|www\\." pattern alone
    misses these, leaving a garbled domain+slug string in the text
    zero-shot classification actually sees. That string is not real
    content, but it can still read as significant to a classifier and
    swing its score -- confirmed case: a link-share post with almost no
    other text ("Read more about the event here: [bare link]") scored
    0.89 for "fire" purely off the slug text, despite the post having
    nothing to do with a disaster. Only a domain immediately followed by a
    "/path" is stripped (not just any "word.word"), so an ordinary
    sentence with no space after a period ("Grabe.Ang lakas") is not
    mistaken for a link.
    """
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\b[a-zA-Z0-9][\w-]*\.[a-zA-Z]{2,24}/\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return " ".join(text.split())


def classify_with_threshold(text, candidate_labels, threshold):
    """
    Independent per-label zero-shot classification (multi_label=True):
    each candidate is scored on its own entailment probability instead of
    a forced ranking across all candidates, so "none of these fit" is a
    real, reachable outcome -- this is what actually fixes the
    forced-choice bug (non-disaster posts getting shoved into a label),
    not a model swap by itself.
    """
    if not text or not text.strip() or not candidate_labels:
        return None, 0.0

    # clean_for_classification can reduce a non-empty post to nothing (a
    # post that's only a URL/@mention, e.g. a link-share bot) -- the check
    # above only guards the pre-cleaned text, so re-check after cleaning
    # too, or the HF pipeline raises "You must include at least one label
    # and at least one sequence" on the empty string.
    cleaned_text = clean_for_classification(text)
    if not cleaned_text:
        return None, 0.0

    classifier = get_zero_shot_classifier()
    result = classifier(cleaned_text, candidate_labels, multi_label=True)

    passing = [
        (label, score) for label, score in zip(result["labels"], result["scores"])
        if score >= threshold
    ]
    if not passing:
        return None, max(result["scores"]) if result["scores"] else 0.0

    best_label, best_score = max(passing, key=lambda pair: pair[1])
    return best_label, best_score
