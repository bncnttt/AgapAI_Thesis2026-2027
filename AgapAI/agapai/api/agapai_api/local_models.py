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

    try:
        detected = langdetect.detect(text)
    except Exception:
        detected = None

    if detected == "en":
        return "eng_Latn"
    if detected == "tl":
        return "tgl_Latn"
    return "ceb_Latn"

CLEARLY_FOREIGN_LANGUAGE_CODES = {
    "pt", "es", "fr", "de", "it", "nl", "ru", "ja", "ko", "zh-cn", "zh-tw",
    "ar", "th", "vi", "tr", "pl", "sv", "el", "id", "ms",
}
FOREIGN_LANGUAGE_CONFIDENCE_THRESHOLD = 0.90


def is_confidently_foreign_language(text):

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

    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\b[a-zA-Z0-9][\w-]*\.[a-zA-Z]{2,24}/\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    return " ".join(text.split())


def classify_with_threshold(text, candidate_labels, threshold):

    if not text or not text.strip() or not candidate_labels:
        return None, 0.0
    
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
