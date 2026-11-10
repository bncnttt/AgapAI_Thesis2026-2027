"""
Dual-condition Cebu disaster post retrieval and filtering.

A fetched post is only persisted into MongoDB 'raw_posts' if it satisfies
ALL of these conditions:

  0. Is not confidently written in a language unrelated to the
     Philippines             (local_models.is_confidently_foreign_language)
  1. Does not name any place in the      (geography.is_out_of_bounds)
     gazetteer's out_of_bounds_blacklist
     (other PH regions, foreign places)
  2. Has Cebu location context           (cascade_check_cebu_relevance)
  3. Is about one of the 5 target         (disaster_relevance.predict_disaster_type)
     disasters (flood, fire, earthquake,
     landslide, typhoon), per a
     multilingual model

Everything that fails any condition -- general Cebu news, crime/police
reports, out-of-scope events, posts about a disaster elsewhere, or content
in an unrelated language -- is discarded. Condition 3 reads the post in its
original language (English, Filipino/Tagalog, Bisaya) with no translation
and no keyword list, and accepts general disaster reporting/discussion: a
post does not need to ask for or offer help.

Condition 0 rejects posts confidently written in an unrelated language
before either content check runs, using the same language-detection gate
the classifier pipeline (pipeline.py) uses.

Condition 1 is a hard veto, checked on the post's own text before Cebu
context is even considered: a post naming an out-of-bounds place (e.g.
"Leyte", "Indonesia") is dropped even if it also happens to name a Cebu
place, and even if the post's AUTHOR is Cebu-based (the 3-level cascade in
cascade_check_cebu_relevance below can otherwise confirm Cebu relevance
from the author's bio/recent posts alone, which says nothing about where
the specific reported disaster actually is).
"""
import re

from agapai_api.bluesky import (
    format_bluesky_date_bound,
    get_attr,
    get_author_bio_text,
    get_recent_post_texts,
    get_reply_parent_uri,
    resolve_incremental_since,
    search_bluesky_posts,
)
from agapai_api.clients import db, mongo_connected
from agapai_api.corpus import get_dialect_only_disaster_terms
from agapai_api.pipeline import DISASTER_TYPES, MIN_SUBSTANTIVE_WORDS, classify, predict_disaster_type
from agapai_api.geography import (
    CEBU_EXPLICIT_MARKERS,
    CEBU_LANDMARK_MARKERS,
    PSGC_CEBU_ALL_LOCATIONS,
    PSGC_CEBU_BARANGAYS,
    find_matching_phrases,
    format_location_name,
    get_cebu_location_search_terms,
    is_out_of_bounds,
)
from agapai_api.local_models import (
    clean_for_classification,
    get_zero_shot_classifier,
    is_confidently_foreign_language,
)

# Minimum model confidence for a post to count as a real disaster report.
# Stricter than pipeline.py's own floor: retrieval would rather miss a
# borderline post than save emotional/poetic noise.
RETRIEVAL_MIN_DISASTER_CONFIDENCE = 0.60

# The zero-shot non-disaster veto only fires when its top non-disaster label
# is this confident. Zero-shot NLI is weak on Tagalog/Bisaya, so a low-scoring
# top label (e.g. 0.26 "municipal news" for a house-fire report) is noise, not
# evidence; genuine non-disaster posts (bomb threat, crash, budget news) score
# well above this.
NON_DISASTER_VETO_MIN_SCORE = 0.60

# Gazetteer place names that are also everyday Tagalog/Bisaya words. A bare
# match on one of these says nothing about location ("talaga" = really,
# "sunog" = burnt/fire, "bagay" = thing, "mga" = plural marker), so it only
# counts when the post structurally marks it as a place, e.g. "Brgy. Bunga",
# "City of Naga" or "Sunog, Cebu".
COMMON_WORD_LOCALITY_VETO = frozenset({
    "talaga", "bagay", "sunog", "daan", "mga", "manga", "sugat", "bato", "bago",
    "bahay", "bala", "bili", "bunga", "bulak", "guba", "gabi", "lupa", "luka",
    "luyo", "lubo", "libo", "lanas", "lamak", "lamesa", "mainit", "maalat",
    "mabasa", "masa", "hagdan", "balud", "agus", "apas", "gawi", "duyan",
    "bugas", "centro", "campo", "poblacion",
})

_LOCALITY_ANCHOR_BEFORE = (
    r"(?:brgy|bgy|barangay|municipality\s+of|city\s+of|town\s+of|sitio|purok|cebu)[\s.,:-]+"
)
_LOCALITY_ANCHOR_AFTER = r"[\s,.-]+(?:cebu|city)\b"

# Pure natural language semantic descriptions for zero-shot classification (no keyword lists)
_DISASTER_EVENT_LABELS = [
    "a flood",
    "a fire",
    "a landslide",
    "a typhoon or storm",
    "an earthquake",
]

_NON_DISASTER_EVENT_LABELS = [
    "a crime, bomb threat or security incident",
    "a road or traffic accident, vehicle crash or someone run over",
    "a protest, strike or police action",
    "government, municipal or business news",
    "ordinary everyday commentary or travel",
]

_ZERO_SHOT_HYPOTHESIS_TEMPLATE = (
    "This text describes {} or emergency assistance for affected victims."
)

_LATIN_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")
_ANY_LETTER = re.compile(r"[^\W\d_]")


def is_allowed_language(text):
    """
    English, Tagalog or Bisaya only. Rejects text that is mostly non-Latin
    script (CJK, Arabic, Cyrillic, Thai, ...), then defers to the shared
    langdetect gate for Latin-script foreign languages. langdetect has no
    Cebuano model, so only *confidently* foreign detections are rejected --
    ambiguous Bisaya must not be thrown away.
    """
    if not text or not text.strip():
        return False

    letters = _ANY_LETTER.findall(text)
    if letters and len(_LATIN_LETTER.findall(text)) / len(letters) < 0.6:
        return False

    return not is_confidently_foreign_language(text)


def is_disaster_related_text(text):
    """
    True when the multilingual model reads the post as being about one of
    the 5 target disasters. Decided by the model alone -- no keyword,
    substring or regex rule. General reporting/discussion counts; the post
    need not ask for or offer help.
    """
    if not text:
        return False
    cleaned_text = clean_for_classification(text)
    if len(cleaned_text.split()) < MIN_SUBSTANTIVE_WORDS:
        return False
    # The model's own "general disaster" class means "a disaster, type
    # unspecified" (e.g. "marami ang nasunogan ... nangangailangan ng damit"
    # scores general disaster 0.79). predict_disaster_type() collapses that to
    # "none", which would drop genuine victim posts, so read the raw label.
    prediction = classify(cleaned_text)
    if (
        prediction["disaster"] not in DISASTER_TYPES + ["general disaster"]
        or prediction["disaster_confidence"] < RETRIEVAL_MIN_DISASTER_CONFIDENCE
    ):
        return False
    return not _is_non_disaster_event(text)


def _is_non_disaster_event(text):
    """
    Second semantic opinion for all posts passing language gating.
    The primary classifier must select one of the disaster classes, so non-disaster
    events (bomb threats, traffic accidents, municipal news) can occasionally yield
    a false high confidence. Here each post is evaluated by zero-shot NLI against
    disaster vs. non-disaster event descriptions without language gating or keywords.
    """
    cleaned_text = clean_for_classification(text)
    if not cleaned_text:
        return False

    result = get_zero_shot_classifier()(
        cleaned_text,
        _DISASTER_EVENT_LABELS + _NON_DISASTER_EVENT_LABELS,
        multi_label=False,
        hypothesis_template=_ZERO_SHOT_HYPOTHESIS_TEMPLATE,
    )
    return (
        result["labels"][0] in _NON_DISASTER_EVENT_LABELS
        and result["scores"][0] >= NON_DISASTER_VETO_MIN_SCORE
    )


def find_valid_cebu_localities(text):
    """
    Gazetteer locality matches in text, minus bare matches on common words
    (COMMON_WORD_LOCALITY_VETO) that are not structurally anchored as a place.
    """
    if not text:
        return []

    valid = []
    for match in find_matching_phrases(text, PSGC_CEBU_ALL_LOCATIONS):
        name = match.lower()
        if name in COMMON_WORD_LOCALITY_VETO:
            escaped = re.escape(name)
            anchored = re.search(
                _LOCALITY_ANCHOR_BEFORE + escaped + r"\b", text, flags=re.IGNORECASE
            ) or re.search(r"\b" + escaped + _LOCALITY_ANCHOR_AFTER, text, flags=re.IGNORECASE)
            if not anchored:
                continue
        valid.append(match)
    return valid


def mentions_cebu_explicitly(text):
    """True when text names Cebu itself (or a Cebu landmark)."""
    if not text:
        return False
    return bool(
        find_matching_phrases(text, CEBU_EXPLICIT_MARKERS)
        or find_matching_phrases(text, CEBU_LANDMARK_MARKERS)
    )


def cascade_check_cebu_relevance(post_text, author_did=None, author_handle=None):
    """
    3-level cascade for confirming Cebu relevance, cheapest check first --
    each level only runs if the prior one fails, to minimize Bluesky API
    usage:

      Level 1: the post's own text names Cebu     (Condition A)
      Level 2: the author's profile bio           (Condition B: valid locality
                                                   in post, Cebu in bio)
      Level 3: the author's 5 most recent posts   (Condition B: "Cebu" in any)

    Exists for posts that name a genuine, specific Cebu place (e.g. a
    barangay) but never say the word "Cebu" itself -- e.g. "may landslide
    sa Pangdan, city of naga" contains two real PSGC locality names
    (Pangdan, Naga). Such a post needs the author's bio/timeline to confirm
    Cebu. Locality matches on common words ("talaga", "bagay", "sunog") are
    ignored unless anchored (see find_valid_cebu_localities), and a post
    with no valid locality and no "Cebu" is dropped.

    Returns (is_relevant, matched_at_level, matches) where
    matched_at_level is one of "post_text", "author_bio",
    "author_recent_posts", or None.

    A hard veto runs first, on the post's own text only: if it names any
    out_of_bounds_blacklist place (e.g. "Leyte", "Indonesia"), the post is
    rejected outright, before any of the 3 levels run. Without this, a
    post entirely about a disaster in another region still passes here as
    long as its AUTHOR is Cebu-based -- Level 2/3 confirm the account, not
    where the reported disaster actually is.
    """
    if is_out_of_bounds(post_text):
        return False, None, []

    # Condition A: the post itself names Cebu (explicitly or via a landmark).
    if mentions_cebu_explicitly(post_text):
        return True, "post_text", find_valid_cebu_localities(post_text)

    # Condition B: a verified locality but no "Cebu" -- accepted only when
    # the author's bio or recent posts confirm Cebu. With no valid locality
    # at all (e.g. only "talaga"/"bagay"/"sunog") there is nothing to
    # verify, so the post is dropped.
    post_localities = find_valid_cebu_localities(post_text)
    if not post_localities:
        return False, None, []

    actor = author_did or author_handle
    if not actor:
        return False, None, []

    bio_text = get_author_bio_text(actor)
    bio_localities = find_valid_cebu_localities(bio_text)
    if mentions_cebu_explicitly(bio_text) or bio_localities:
        return True, "author_bio", sorted(set(post_localities + bio_localities))

    for recent_text in get_recent_post_texts(actor, limit=5):
        if mentions_cebu_explicitly(recent_text):
            return True, "author_recent_posts", post_localities

    return False, None, []


def post_matches_dual_condition(post_text, author_did=None, author_handle=None):
    """
    Valid Post = [English/Tagalog/Bisaya] AND [About one of the 5 target
    disasters, per the multilingual model] AND [Does not name an
    out-of-bounds place] AND [Cebu confirmed via the 3-level cascade].

    Ordered cheapest-first: language, the local model, then the cascade
    (which includes the out-of-bounds veto and may call Bluesky for the
    author's bio/recent posts). Returns (is_cebu_post, is_disaster_related);
    is_cebu_post is only evaluated once the post is disaster-related.
    """
    if not post_text:
        return False, False

    if not is_allowed_language(post_text):
        return False, False

    if not is_disaster_related_text(post_text):
        return False, False

    is_cebu_post, _level, _matches = cascade_check_cebu_relevance(
        post_text, author_did=author_did, author_handle=author_handle
    )
    return is_cebu_post, True


def retrieve_and_filter_cebu_disaster_posts(
    start_date,
    end_date,
    max_posts=-1,
    location_terms=None,
):
    """
    Retrieves Cebu posts via dynamic PSGC location-anchored search, then
    applies the strict dual-condition filter before persisting matching
    posts to MongoDB 'raw_posts' (upserted by URI).
    """
    if not mongo_connected or db is None:
        raise RuntimeError("MongoDB is not connected.")

    location_terms = location_terms if location_terms is not None else get_cebu_location_search_terms()

    since_value = format_bluesky_date_bound(start_date)
    until_value = format_bluesky_date_bound(end_date, end_of_day=True)

    raw_posts_col = db["raw_posts"]

    # Dedup against what's already saved, fetched once up front so each
    # candidate post is a cheap in-memory lookup instead of a per-post DB
    # round trip.
    existing_uris = set(raw_posts_col.distinct("uri"))
    skipped_already_saved = 0

    # True incremental retrieval: ask Bluesky only for what's newer than
    # the latest post we've already saved to raw_posts, instead of
    # re-fetching the full requested range every run. Falls back to the
    # originally requested since_value when raw_posts is empty (first run).
    bluesky_since_value = resolve_incremental_since(raw_posts_col, fallback_since=since_value)
    used_incremental_fetch = bluesky_since_value != since_value
    print(
        f"[AGAPAI LOG] {'Incremental' if used_incremental_fetch else 'Full historical'} "
        f"fetch: querying Bluesky since {bluesky_since_value}."
    )

    total_fetched = 0
    matched_count = 0
    discarded_count = 0
    seen_uris = set()

    for search_query in location_terms:
        try:
            search_results = search_bluesky_posts(
                search_query, max_posts, since=bluesky_since_value, until=until_value
            )
        except RuntimeError as search_error:
            print(f"[AGAPAI LOG] WARNING: search failed for '{search_query}': {search_error}")
            continue

        for post_view in search_results:
            post_uri = get_attr(post_view, "uri")
            if not post_uri or post_uri in seen_uris:
                continue

            if post_uri in existing_uris:
                seen_uris.add(post_uri)
                skipped_already_saved += 1
                continue
            seen_uris.add(post_uri)
            total_fetched += 1

            record = get_attr(post_view, "record")
            post_text = get_attr(record, "text", "") if record else get_attr(post_view, "text", "")

            author = get_attr(post_view, "author")
            is_cebu_post, is_disaster_related = post_matches_dual_condition(
                post_text,
                author_did=get_attr(author, "did"),
                author_handle=get_attr(author, "handle"),
            )

            if not (is_cebu_post and is_disaster_related):
                discarded_count += 1
                continue

            matched_count += 1
            raw_post_document = {
                "uri": post_uri,
                "author_did": get_attr(author, "did"),
                "author_handle": get_attr(author, "handle"),
                "text": post_text,
                "reply_parent_uri": get_reply_parent_uri(post_view),
                "created_at": get_attr(record, "created_at") if record else None,
                "retrieval_source": "dual_condition_cebu_disaster_filter",
                "date_range_start": start_date,
                "date_range_end": end_date,
            }
            raw_posts_col.update_one({"uri": post_uri}, {"$set": raw_post_document}, upsert=True)

    print(f"[AGAPAI LOG] Total Cebu posts fetched: {total_fetched}")
    print(f"[AGAPAI LOG] Posts matching 5 target disasters: {matched_count}")
    print(f"[AGAPAI LOG] Discarded non-disaster / out-of-scope posts: {discarded_count}")
    print(f"[AGAPAI LOG] Skipped {skipped_already_saved} already-saved posts (deduped by URI).")

    return {
        "total_fetched": total_fetched,
        "matched": matched_count,
        "discarded": discarded_count,
        "skipped_already_saved": skipped_already_saved,
        "bluesky_since_value": bluesky_since_value,
        "used_incremental_fetch": used_incremental_fetch,
    }


def retrieve_and_cascade_filter_disaster_posts(
    start_date,
    end_date,
    max_posts=-1,
    include_barangays=False,
):
    """
    Complements retrieve_and_filter_cebu_disaster_posts(): that function
    anchors every search query with a Cebu place name + the literal word
    "Cebu", so it structurally cannot find a genuine post like "may
    landslide sa Pangdan, city of naga" that names a real Cebu barangay
    and municipality but never says "Cebu" at all.

    This searches Bluesky using ONLY the dynamically generated
    multilingual disaster terms (no location anchor whatsoever), then
    confirms Cebu relevance per-candidate via the 3-level cascade
    (cascade_check_cebu_relevance) instead of requiring "Cebu" in the
    post text.

    The actual Bluesky search queries use ONLY the Tagalog/Cebuano dialect
    subset of the corpus, never the English WordNet-expanded terms: a
    bare English disaster word (e.g. "blaze", "fire") is also common,
    unrelated global vocabulary, so searching it with no location anchor
    at all pulled in large volumes of irrelevant global content (verified
    case: a UK "New Forest wildfire" post surfaced from a bare "blaze"
    search). Terms shorter than 4 characters are excluded from search
    queries -- with no location anchor at all, a very short root term
    would still return an unusably large volume of content.

    Those terms only decide what Bluesky is asked for. Whether a fetched
    candidate is actually about one of the 5 target disasters is decided by
    the multilingual model (is_disaster_related_text), not by matching
    those terms against the post.
    """
    if not mongo_connected or db is None:
        raise RuntimeError("MongoDB is not connected.")

    search_source_terms = get_dialect_only_disaster_terms()

    since_value = format_bluesky_date_bound(start_date)
    until_value = format_bluesky_date_bound(end_date, end_of_day=True)

    raw_posts_col = db["raw_posts"]
    existing_uris = set(raw_posts_col.distinct("uri"))
    skipped_already_saved = 0

    bluesky_since_value = resolve_incremental_since(raw_posts_col, fallback_since=since_value)
    used_incremental_fetch = bluesky_since_value != since_value
    print(
        f"[AGAPAI LOG] {'Incremental' if used_incremental_fetch else 'Full historical'} "
        f"cascade fetch: querying Bluesky since {bluesky_since_value}."
    )

    total_fetched = 0
    matched_count = 0
    discarded_count = 0
    seen_uris = set()

    search_terms = sorted({term for term in search_source_terms if len(term) >= 4})

    if include_barangays:
        # Bare (unanchored) barangay names, e.g. "Tabunok", "Pangdan": no
        # location term at all requires the literal word "Cebu" (the
        # anchored search in geography.py) or a specific disaster-term
        # match in Bluesky's own search index (which does exact-token
        # matching, not stemming -- confirmed live: searching "sunog"
        # never matches a post that only says "nasunog"). A barangay name
        # is, unlike a generic disaster word, inherently distinctive
        # enough to search bare with low noise (verified live: "Tabunok"
        # and "Pangdan" returned 8 and 3 total global results
        # respectively, each including the target post) -- Cebu relevance
        # is still confirmed downstream by the cascade, same as every
        # other candidate here.
        barangay_terms = {format_location_name(name) for name in PSGC_CEBU_BARANGAYS}
        search_terms = sorted(set(search_terms) | barangay_terms)

    for search_query in search_terms:
        try:
            search_results = search_bluesky_posts(
                search_query, max_posts, since=bluesky_since_value, until=until_value
            )
        except RuntimeError as search_error:
            print(f"[AGAPAI LOG] WARNING: search failed for '{search_query}': {search_error}")
            continue

        for post_view in search_results:
            post_uri = get_attr(post_view, "uri")
            if not post_uri or post_uri in seen_uris:
                continue
            if post_uri in existing_uris:
                seen_uris.add(post_uri)
                skipped_already_saved += 1
                continue
            seen_uris.add(post_uri)
            total_fetched += 1

            record = get_attr(post_view, "record")
            post_text = get_attr(record, "text", "") if record else get_attr(post_view, "text", "")
            author = get_attr(post_view, "author")
            author_did = get_attr(author, "did")
            author_handle = get_attr(author, "handle")

            # Reject on language first, before spending a cascade check (up
            # to 2 extra Bluesky calls for bio/timeline) or a model pass on
            # content that can't be genuinely Cebuano/Tagalog/English.
            if not is_allowed_language(post_text):
                discarded_count += 1
                continue

            # The model runs locally, so it goes before the cascade: a post
            # that isn't about one of the 5 disasters never costs the
            # cascade's Bluesky calls.
            if not is_disaster_related_text(post_text):
                discarded_count += 1
                continue

            is_cebu_relevant, _matched_level, _matches = cascade_check_cebu_relevance(
                post_text, author_did=author_did, author_handle=author_handle
            )
            if not is_cebu_relevant:
                discarded_count += 1
                continue

            matched_count += 1
            raw_post_document = {
                "uri": post_uri,
                "author_did": author_did,
                "author_handle": author_handle,
                "text": post_text,
                "reply_parent_uri": get_reply_parent_uri(post_view),
                "created_at": get_attr(record, "created_at") if record else None,
                "retrieval_source": "cascade_disaster_term_filter",
                "date_range_start": start_date,
                "date_range_end": end_date,
            }
            raw_posts_col.update_one({"uri": post_uri}, {"$set": raw_post_document}, upsert=True)

    print(f"[AGAPAI LOG] Cascade search: total fetched {total_fetched}")
    print(f"[AGAPAI LOG] Cascade search: matched {matched_count}")
    print(f"[AGAPAI LOG] Cascade search: discarded {discarded_count}")
    print(f"[AGAPAI LOG] Cascade search: skipped {skipped_already_saved} already-saved posts.")

    return {
        "total_fetched": total_fetched,
        "matched": matched_count,
        "discarded": discarded_count,
        "skipped_already_saved": skipped_already_saved,
        "bluesky_since_value": bluesky_since_value,
        "used_incremental_fetch": used_incremental_fetch,
    }