import nltk
from functools import lru_cache
from nltk.corpus import wordnet

from agapai_api.config import load_local_env

load_local_env()

# Existing English search/expansion seeds.
DISASTER_CONCEPT_SEEDS = ["fire", "flood", "earthquake", "landslide", "typhoon"]

# Retrieval vocabulary only; these rules never enter model inference.
CEBUANO_RETRIEVAL_KEYWORDS = {
    "fire": ("sunog", "nasunog", "nasunogan", "kasunogan"),
    "flood": ("baha", "nagbaha", "gibaha", "gibahaan"),
    "earthquake": ("linog", "milinog", "naglinog", "paglinog"),
    "landslide": ("pagdahili sa yuta", "midahili ang yuta", "nidahili ang yuta", "nahugno ang yuta"),
    "typhoon": ("bagyo", "bagyong", "gibagyo"),
}

FILIPINO_RETRIEVAL_KEYWORDS = {
    "fire": ("sunog", "nasunog", "nasunugan", "pagsunog"),
    "flood": ("baha", "binaha", "nabaha", "nabahaan", "pagbaha", "bumaha"),
    "earthquake": ("lindol", "lumindol", "pagyanig"),
    "landslide": ("pagguho ng lupa", "gumuhong lupa"),
    "typhoon": ("bagyo", "binagyo"),
}


@lru_cache(maxsize=1)
def get_retrieval_keyword_groups():
    """Retain the existing English/dialect expansion, grouped by its seed."""
    return {
        seed: set(get_multilingual_disaster_terms([seed])) | {seed} | set(words)
              | set(FILIPINO_RETRIEVAL_KEYWORDS[seed])
        for seed, words in CEBUANO_RETRIEVAL_KEYWORDS.items()
    }


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


@lru_cache(maxsize=4096)
def _translate_search_term(term, target_language_code):
    """Reuse unchanged search-vocabulary translations across seeds and runs."""
    from agapai_api.local_models import translate_text
    return translate_text(term, "eng_Latn", target_language_code)


def request_dialect_expansion_locally(english_terms):

    dialect_terms = set()
    for term in english_terms:
        for target_language_code in ("tgl_Latn", "ceb_Latn"):
            translated = _translate_search_term(term, target_language_code)
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

    english_terms = expand_disaster_concepts_via_wordnet(seed_terms)
    return sorted(set(request_dialect_expansion_locally(english_terms)) | DIALECT_SEARCH_INFLECTIONS
                  | {term for groups in (CEBUANO_RETRIEVAL_KEYWORDS, FILIPINO_RETRIEVAL_KEYWORDS)
                     for words in groups.values() for term in words})


def get_multilingual_disaster_terms(seed_terms=None):

    english_terms = expand_disaster_concepts_via_wordnet(seed_terms)
    dialect_terms = request_dialect_expansion_locally(english_terms)

    selected = seed_terms if seed_terms is not None else DISASTER_CONCEPT_SEEDS
    explicit = {term for seed in selected for term in CEBUANO_RETRIEVAL_KEYWORDS.get(seed, ())}
    return sorted(set(english_terms) | set(dialect_terms) | explicit)
