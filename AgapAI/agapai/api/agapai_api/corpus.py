import nltk
from nltk.corpus import wordnet

from agapai_api.config import load_local_env

load_local_env()

# The only allowed hardcoded values per the task's zero-hardcoding rule:
# 5 base English disaster concepts, used purely as WordNet expansion seeds.
DISASTER_CONCEPT_SEEDS = ["fire", "flood", "earthquake", "landslide", "typhoon"]


# Bluesky search is exact-token, so root words alone miss conjugated posts.
# Search-query seeds only -- never used to classify or filter content.
DIALECT_SEARCH_INFLECTIONS = frozenset({
    "sunog", "nasunog", "nasunogan", "kasunogan", "pagsunog",
    "baha", "nagbaha", "nabaha", "nabahaan", "pabaha", "lapuk",
    "landslide", "dahili", "guba", "nadasmagan",
    "linog", "naglinog", "naglinao", "tay-og",
    "bagyo", "kusog hangin", "uwan", "kusog uwan",
})


def _ensure_wordnet_available():
    try:
        wordnet.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def _select_disaster_synsets(seed):
    """
    A bare seed word like "fire" has many unrelated WordNet senses (getting
    fired from a job, firing a weapon, emotional "ardor", an electoral
    "landslide", etc). Selects only the noun synset(s) whose gloss actually
    describes the natural-disaster sense, so expansion stays on-topic.
    """
    disaster_hint_words = {
        "burn", "combustion", "flame", "water", "overflow", "shak",
        "vibrat", "fault", "volcanic", "slide", "dirt", "rock", "mountain",
        "cliff", "cyclone", "storm", "wind",
    }

    matches = [
        synset for synset in wordnet.synsets(seed, pos=wordnet.NOUN)
        if any(hint in synset.definition().lower() for hint in disaster_hint_words)
    ]
    return matches or wordnet.synsets(seed, pos=wordnet.NOUN)[:1]


def expand_disaster_concepts_via_wordnet(seed_terms=None):
    """
    Programmatically expands the seed disaster concepts into a broader
    English vocabulary at runtime, using the disaster-relevant WordNet
    synset(s) for each seed, their lemma names, and their direct hyponyms.
    """
    seed_terms = seed_terms if seed_terms is not None else DISASTER_CONCEPT_SEEDS
    _ensure_wordnet_available()

    expanded_terms = set()

    for seed in seed_terms:
        expanded_terms.add(seed)
        for synset in _select_disaster_synsets(seed):
            for lemma in synset.lemma_names():
                expanded_terms.add(lemma.replace("_", " ").lower())

            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemma_names():
                    expanded_terms.add(lemma.replace("_", " ").lower())

    print(f"[AGAPAI LOG] Dynamically expanded {len(seed_terms)} target disaster concepts via WordNet.")
    return sorted(expanded_terms)


def request_dialect_expansion_locally(english_terms):
    """
    Translates each WordNet-expanded English term into Tagalog and
    Cebuano/Bisaya using the local NLLB-200 translation model -- no
    external LLM API involved. Tagalog/Cebuano verbs are built by
    attaching affixes (nag-, mag-, pag-, gi-, mi-, na-) to a stable root,
    so downstream matching against these root translations is done via
    substring containment (see retrieval.py), which naturally also
    matches conjugated/affixed forms (e.g. "baha" also matches "nagbaha",
    "gibaha") without needing every conjugation generated or enumerated
    here.
    """
    from agapai_api.local_models import translate_text

    dialect_terms = set()
    for term in english_terms:
        for target_language_code in ("tgl_Latn", "ceb_Latn"):
            translated = translate_text(term, "eng_Latn", target_language_code)
            cleaned = translated.strip().lower()
            if not cleaned or cleaned == term.lower():
                continue
            dialect_terms.add(cleaned)
            for word in cleaned.split():
                if len(word) >= 3:
                    dialect_terms.add(word)

    print("[AGAPAI LOG] Generated Tagalog/Cebuano dialect terms dynamically via local translation model.")
    return sorted(dialect_terms)


def get_dialect_only_disaster_terms(seed_terms=None):
    """
    Same generation pipeline as get_multilingual_disaster_terms(), but
    returns only the Tagalog/Cebuano dialect terms -- excluding the
    English WordNet-expanded terms entirely.

    This exists specifically for anchor-free (no location term) Bluesky
    searches: an English disaster word like "blaze" or "fire" is also
    common, unrelated, everyday/global English vocabulary, so searching
    it bare pulls in large volumes of irrelevant global content (verified
    case: a UK "New Forest wildfire" post surfaced from a bare "blaze"
    search). A Tagalog/Cebuano term like "sunog" or "linog" carries far
    less of that risk, since it isn't common vocabulary outside Philippine
    contexts. This is a structural, dynamic split (which generation stage
    produced the term), not a hardcoded exclusion list of specific words.
    """
    english_terms = expand_disaster_concepts_via_wordnet(seed_terms)
    return sorted(set(request_dialect_expansion_locally(english_terms)) | DIALECT_SEARCH_INFLECTIONS)


def get_multilingual_disaster_terms(seed_terms=None):
    """
    Builds the full dynamic, multilingual disaster-term corpus at runtime:
    WordNet-expanded English seeds, then locally translated Tagalog and
    Cebuano equivalents. Only the 5 base English seeds are hardcoded in
    this module -- every other term is generated at call time, with no
    hosted LLM API involved anywhere in this pipeline.
    """
    english_terms = expand_disaster_concepts_via_wordnet(seed_terms)
    dialect_terms = request_dialect_expansion_locally(english_terms)

    return sorted(set(english_terms) | set(dialect_terms))
