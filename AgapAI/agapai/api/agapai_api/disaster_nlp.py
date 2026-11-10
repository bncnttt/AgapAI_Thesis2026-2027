"""
Local, multilingual translation + zero-shot classification pipeline for
Cebu disaster posts.

Translation: Meta's NLLB-200 (distilled 600M), run entirely locally, covers
Tagalog and Cebuano/Bisaya natively (unlike Multilingual WordNet/OMW-1.4,
which has fragmented Philippine-language coverage and can't handle
colloquial code-switched text at all). Taglish/Bislish code-switched posts
are routed through NLLB using the detected dominant language; NLLB's own
robustness to embedded English loanwords handles the code-switching itself.

Classification: zero-shot NLI classification (bart-large-mnli) against the
5 target disaster types, then against the 2 help-intent categories -- no
manually labeled training set required for either step.
"""
import langdetect
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"
ZERO_SHOT_MODEL_NAME = "facebook/bart-large-mnli"

DISASTER_TYPE_LABELS = ["flood", "earthquake", "typhoon", "fire", "landslide"]
NOT_A_DISASTER_LABEL = "not a disaster"
HELP_INTENT_LABELS = ["needing help", "willing to help"]

DISASTER_TYPE_CONFIDENCE_THRESHOLD = 0.4
HELP_INTENT_CONFIDENCE_THRESHOLD = 0.35

_device = 0 if torch.cuda.is_available() else -1
_translation_tokenizer = None
_translation_model = None
_zero_shot_classifier = None


def _load_translation_model():
    global _translation_tokenizer, _translation_model
    if _translation_model is None:
        print("[AGAPAI LOG] Loading local NLLB-200 translation model (first call only)...")
        _translation_tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        _translation_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
    return _translation_tokenizer, _translation_model


def _load_zero_shot_classifier():
    global _zero_shot_classifier
    if _zero_shot_classifier is None:
        print("[AGAPAI LOG] Loading local zero-shot classification model (first call only)...")
        _zero_shot_classifier = pipeline(
            "zero-shot-classification", model=ZERO_SHOT_MODEL_NAME, device=_device
        )
    return _zero_shot_classifier


def detect_source_language_code(text):
    """
    Guesses an NLLB-200 source-language code for the given text at runtime.
    langdetect doesn't have a distinct Cebuano class, so anything not
    confidently detected as English or Tagalog defaults to Cebuano -- the
    dominant native language in Cebu Province, this project's area of
    interest. This is a language-routing decision, not a hardcoded
    Tagalog/Cebuano vocabulary list.
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


def translate_to_english(text):
    """
    Translates Tagalog / Cebuano / code-switched (Taglish, Bislish) text to
    English via the local NLLB-200 model. Returns the text unchanged if
    it's already detected as English.
    """
    if not text or not text.strip():
        return text

    source_lang = detect_source_language_code(text)
    if source_lang == "eng_Latn":
        return text

    tokenizer, model = _load_translation_model()
    tokenizer.src_lang = source_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded, forced_bos_token_id=forced_bos_token_id, max_length=256
        )

    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated[0] if translated else text


def classify_disaster_type(english_text):
    """
    Zero-shot classifies the (already translated) text strictly against the
    5 target disaster types plus a "not a disaster" rejection label.
    Returns (disaster_type_or_None, confidence).
    """
    if not english_text or not english_text.strip():
        return None, 0.0

    classifier = _load_zero_shot_classifier()
    result = classifier(english_text, DISASTER_TYPE_LABELS + [NOT_A_DISASTER_LABEL])

    top_label = result["labels"][0]
    top_score = float(result["scores"][0])

    if top_label == NOT_A_DISASTER_LABEL or top_score < DISASTER_TYPE_CONFIDENCE_THRESHOLD:
        return None, top_score

    return top_label, top_score


def classify_help_intent(english_text):
    """
    Zero-shot classifies help intent. Only meaningful for text already
    confirmed to be one of the 5 target disaster types. Returns
    (intent_or_None, confidence).
    """
    if not english_text or not english_text.strip():
        return None, 0.0

    classifier = _load_zero_shot_classifier()
    result = classifier(english_text, HELP_INTENT_LABELS)

    top_label = result["labels"][0]
    top_score = float(result["scores"][0])

    if top_score < HELP_INTENT_CONFIDENCE_THRESHOLD:
        return None, top_score

    return top_label, top_score


def classify_disaster_post(raw_text):
    """
    Full pipeline for one post: translate to English if needed, then
    strictly classify against the 5 target disaster types, then classify
    help intent only for posts confirmed as one of those 5 types.
    """
    translated_text = translate_to_english(raw_text)
    disaster_type, disaster_confidence = classify_disaster_type(translated_text)

    help_intent = None
    help_confidence = 0.0
    if disaster_type is not None:
        help_intent, help_confidence = classify_help_intent(translated_text)

    # Only report a translation when the source text actually wasn't
    # already English -- otherwise translated_text is just a copy of the
    # original post, which duplicates the "text" field for no reason.
    was_translated = translated_text != raw_text

    return {
        "translated_text": translated_text if was_translated else None,
        "disaster_type": disaster_type,
        "disaster_type_confidence": round(disaster_confidence, 4),
        "help_intent": help_intent,
        "help_intent_confidence": round(help_confidence, 4),
        "is_valid_disaster_post": disaster_type is not None,
    }
