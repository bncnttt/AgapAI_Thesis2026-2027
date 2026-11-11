"""Cebu disaster retrieval: eligibility, author-context cascade, persistence and tracing."""
import json
import logging
import os
import re
from collections import Counter
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

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
from agapai_api.corpus import (
    DISASTER_CONCEPT_SEEDS,
    get_dialect_only_disaster_terms,
    get_retrieval_keyword_groups,
)
from agapai_api.geography import (
    CEBU_EXPLICIT_MARKERS,
    CEBU_LANDMARK_MARKERS,
    PSGC_CEBU_ALL_LOCATIONS,
    PSGC_CEBU_BARANGAYS,
    PSGC_CEBU_CITIES_MUNICIPALITIES,
    find_out_of_bounds_matches,
    format_location_name,
    get_cebu_location_search_terms,
    load_psgc_cebu_gazetteer,
)


logger = logging.getLogger("agapai.retrieval")
_active_retrieval_run = ContextVar("agapai_retrieval_run", default=None)


class _RunContextFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "retrieval_run"):
            record.retrieval_run = _active_retrieval_run.get()
        return True


class _RunFileFilter(logging.Filter):
    def __init__(self, run_id):
        super().__init__()
        self.run_id = run_id

    def filter(self, record):
        return getattr(record, "retrieval_run", None) == self.run_id


# A dedicated handler leaves NER/classifier/root logging unchanged.
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    # Detailed terminal tracing is enabled by default. Explicitly setting 0
    # quiets the console while the run file still records every trace event.
    handler.setLevel(logging.DEBUG if os.getenv("AGAPAI_RETRIEVAL_DEBUG", "1").lower()
                     in {"1", "true", "yes"} else logging.INFO)
    logger.addHandler(handler)
logger.addFilter(_RunContextFilter())
logger.setLevel(logging.DEBUG)
logger.propagate = False
RETRIEVAL_RULE_VERSION = 3

COMMON_PLACE_WORDS = frozenset({
    "talaga", "bagay", "sunog", "daan", "mga", "manga", "sugat", "bato", "bago",
    "bahay", "bala", "bili", "bunga", "bulak", "guba", "gabi", "lupa", "luka",
    "luyo", "lubo", "libo", "lanas", "lamak", "lamesa", "mainit", "maalat",
    "mabasa", "masa", "hagdan", "balud", "agus", "apas", "gawi", "duyan",
    "bugas", "centro", "campo", "poblacion",
})
NON_EVENT = re.compile(
    r"\bfire\s+(?:records?|departments?|stations?|engines?|trucks?|drills?|safety|"
    r"prevention|insurance|sales?|emblem)\b|\blandslide\s+(?:victory|win)\b|"
    r"\bflood\s+of\s+(?:emails?|messages?|notifications?)\b|\bbaha[’']i\b",
    re.IGNORECASE,
)


def normalize(text):
    return re.sub(r"\s+", " ", text or "").strip()


def location_disaster_queries():
    """Bounded municipality-level discovery without requiring literal Cebu.

    Do not multiply the entire barangay registry by the expanded corpus.
    Existing dialect-only queries also discover barangay-only reports.
    """
    return sorted({f"{kind} {place}" for kind in DISASTER_CONCEPT_SEEDS
                   for place in PSGC_CEBU_CITIES_MUNICIPALITIES})


@lru_cache(maxsize=32)
def phrase_pattern(phrases):
    parts = [re.escape(p).replace(r"\ ", r"\s+") for p in phrases if p]
    return re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)", re.IGNORECASE)


def matches(text, phrases):
    names = tuple(sorted(phrases, key=lambda p: (-len(p), p)))
    if not names:
        return []
    return sorted({normalize(m.group()).lower() for m in phrase_pattern(names).finditer(text)})


def disaster_matches(text):
    event_text = NON_EVENT.sub(" ", re.sub(r"https?://\S+", " ", normalize(text)))
    return {kind: found for kind, terms in get_retrieval_keyword_groups().items()
            if (found := matches(event_text, terms))}


def locality_matches(text):
    found = matches(text, PSGC_CEBU_ALL_LOCATIONS)
    return [name for name in found if name not in COMMON_PLACE_WORDS or re.search(
        r"\b(?:brgy|bgy|barangay|city of|municipality of|sitio|purok)[\s.,:-]+"
        + re.escape(name) + r"\b|\b" + re.escape(name) + r"[\s,.-]+(?:cebu|city)\b",
        text, re.IGNORECASE)]


def explicit_matches(text):
    return matches(text, CEBU_EXPLICIT_MARKERS | CEBU_LANDMARK_MARKERS)


def conflicting_location(text, locations):
    if find_out_of_bounds_matches(text):
        return True
    # The current gazetteer has no blacklist section. A shared name with
    # an explicit, unresolved address qualifier must not be rescued by bio.
    for name in locations:
        address = re.search(r"\b(?i:" + re.escape(name) + r"),\s*([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2})", text)
        if address:
            qualifier = address.group(1)
            if not (explicit_matches(qualifier) or locality_matches(qualifier)
                    or qualifier.lower() in {"philippines", "ph"}):
                return True
    # Resolve explicitly named event destinations against the same registry.
    # This also prevents a Cebu reporting/source mention rescuing an event
    # stated to be 'in Japan', without adding a separate geographic list.
    event = re.search(
        r"\b(?i:fire|flood|earthquake|landslide|typhoon|sunog|baha|linog|bagyo)\b"
        r"[^.!?\n]{0,45}?\b(?:in|at|sa)\s+([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2})", text)
    if event:
        place = event.group(1)
        return not (explicit_matches(place) or locality_matches(place))
    return False


@lru_cache(maxsize=1)
def hierarchy_pairs():
    gazetteer = load_psgc_cebu_gazetteer()
    return [(name.lower(), str(row.get("municipality", "")).lower())
            for name, rows in gazetteer.get("barangays", {}).items() for row in rows]


def credible_context(text, recent=False):
    """Resolve profile geography, rather than accepting any shared place name."""
    text = re.sub(r"https?://\S+", " ", text or "")
    if not text or conflicting_location(text, locality_matches(text)):
        return False
    explicit = explicit_matches(text)
    if explicit:
        # A location/address or locative statement links the author to Cebu;
        # 'I love Cebu (City)' alone does not establish that link.
        locative = bool(re.search(
            r"\b(?:in|at|from|sa|taga|based in|live in|located in)\s+(?:\w+\s+){0,3}"
            r"(?:cebu|sugbo)\b|\b(?:cebu|sugbo)[ -]+based\b",
            text, re.IGNORECASE))
        address = bool(re.fullmatch(
            r"(?:cebu|sugbo)(?:\s+(?:city|province))?(?:,?\s*philippines)?[.!]?", text.strip(), re.IGNORECASE))
        resolved_address = bool(locality_matches(text) and re.search(
            r"\b(?:brgy|barangay|sitio|purok)\b.*\b(?:cebu|sugbo)\b", text, re.IGNORECASE))
        return locative or address or resolved_address
    localities = set(locality_matches(text))
    # Reuse the gazetteer's actual barangay -> municipality hierarchy.
    return any(b != m and b in localities and m in localities for b, m in hierarchy_pairs())


def retrieval_metadata(decision):
    """Additive provenance, separate from the existing classifier prediction."""
    return {"retrieval_status": "accepted", "retrieval_rule_version": RETRIEVAL_RULE_VERSION,
            "cebu_matched_level": decision["verification"],
            "matched_disaster_keywords": decision["disasters"],
            "retrieval_disaster_types": list(decision["disasters"])}


class RetrievalRun:
    """Per-run cache, counts and tracing with post details for each decision."""
    def __init__(self, bio_fetch, recent_fetch):
        self.id = uuid4().hex[:10]
        self.bio_fetch, self.recent_fetch = bio_fetch, recent_fetch
        self.bios, self.recent = {}, {}
        self.counts = Counter()
        self.closed = False
        self._file_handler = None
        self.log_path = None
        self._context_token = _active_retrieval_run.set(self.id)
        try:
            log_dir = Path(os.getenv("AGAPAI_RETRIEVAL_LOG_DIR") or
                           Path(__file__).resolve().parents[1] / "logs" / "retrieval")
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
            self.log_path = (log_dir / f"retrieval_{timestamp}_{self.id}.log").resolve()
            self._file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
            self._file_handler.setLevel(logging.DEBUG)
            self._file_handler.setFormatter(logging.Formatter("%(message)s"))
            self._file_handler.addFilter(_RunFileFilter(self.id))
            logger.addHandler(self._file_handler)
        except OSError as error:
            self.log_path = None
            self.error("OPEN_LOG_FILE", error)
        self.log("START_RETRIEVAL", detail=False,
                 log_file=str(self.log_path) if self.log_path else None)

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.error("RETRIEVAL", error)
        self.finish()

    def log(self, stage, detail=True, **fields):
        message = "[RETRIEVAL_TRACE] run=%s %s %s"
        (logger.debug if detail else logger.info)(
            message, self.id, stage, json.dumps(fields, ensure_ascii=True, sort_keys=True),
            extra={"retrieval_run": self.id})

    def query(self, query, count=None):
        self.log("SEARCH_QUERY", query=query, candidates_returned=count)

    def error(self, stage, error, post_id=None):
        self.counts["errors"] += 1
        # Exception type only: provider messages can contain request credentials.
        self.log("ERROR", detail=False, failed_stage=stage, post_id=post_id, error=type(error).__name__)

    def context(self, actor, stage, post_id, available_bio=None):
        cache = self.bios if stage == "AUTHOR_BIO" else self.recent
        if actor not in cache:
            try:
                if stage == "AUTHOR_BIO":
                    text = available_bio if available_bio is not None else self.bio_fetch(actor)
                    cache[actor] = credible_context(text)
                else:
                    cache[actor] = any(credible_context(text, recent=True)
                                       for text in list(self.recent_fetch(actor, limit=5))[:5])
            except Exception as error:
                self.error(stage, error, post_id)
                cache[actor] = False
        self.log("CASCADE_LEVEL=" + stage, post_id=post_id, cebu_verified=cache[actor])
        return cache[actor]

    def evaluate(self, text, post_id=None, actor=None, available_bio=None, *,
                 author_did=None, author_handle=None, search_query=None):
        original_text = text
        text = normalize(re.sub(r"https?://\S+", " ", text or ""))
        self.counts["candidates"] += 1
        self.log("CANDIDATE_FOUND", post_id=post_id)
        disaster = disaster_matches(text)
        self.log("DISASTER_CHECK", post_id=post_id, matched=bool(disaster), matches=disaster)
        locations, explicit, verification = [], [], None
        reason = "NO_DISASTER_MATCH"
        if disaster:
            locations, explicit = locality_matches(text), explicit_matches(text)
            self.log("LOCATION_CHECK", post_id=post_id, source="CEBU_GAZETTEER",
                     locations=locations, explicit_cebu=bool(explicit))
            if conflicting_location(text, locations):
                reason = "CONFLICTING_LOCATION"
            elif not locations and not explicit:
                reason = "NO_GAZETTEER_LOCATION"
            elif explicit:
                verification = "EXPLICIT_CEBU"
            elif actor and self.context(actor, "AUTHOR_BIO", post_id, available_bio):
                verification = "AUTHOR_BIO"
            elif actor and self.context(actor, "AUTHOR_RECENT_POSTS", post_id):
                verification = "AUTHOR_RECENT_POSTS"
            else:
                reason = "CEBU_NOT_VERIFIED"
        accepted = verification is not None
        if accepted:
            reason = "DISASTER_AND_VERIFIED_CEBU_LOCATION"
            self.counts["accepted"] += 1
            self.counts[verification] += 1
        else:
            self.counts["rejected"] += 1
            self.counts[reason] += 1
        post_details = {
            "author_did": author_did or (actor if actor and actor.startswith("did:") else None),
            "author_handle": author_handle or (actor if actor and not actor.startswith("did:") else None),
            "text": original_text,
            "search_query": search_query,
        }
        self.log("DECISION=" + ("ACCEPT" if accepted else "REJECT"), post_id=post_id,
                 reason=reason, verification=verification, **post_details)
        return {"accepted": accepted, "reason": reason, "verification": verification,
                "locations": sorted(set(locations + explicit)), "disasters": disaster}

    def finish(self):
        if not self.closed:
            self.closed = True
            try:
                self.log("RETRIEVAL SUMMARY", detail=False, candidates_found=self.counts["candidates"],
                         accepted=self.counts["accepted"], rejected=self.counts["rejected"],
                         rejected_no_disaster=self.counts["NO_DISASTER_MATCH"],
                         rejected_no_cebu_location=self.counts["NO_GAZETTEER_LOCATION"],
                         rejected_cebu_unverified=self.counts["CEBU_NOT_VERIFIED"],
                         rejected_conflicting=self.counts["CONFLICTING_LOCATION"],
                         verified_explicit=self.counts["EXPLICIT_CEBU"],
                         verified_bio=self.counts["AUTHOR_BIO"],
                         verified_recent=self.counts["AUTHOR_RECENT_POSTS"], errors=self.counts["errors"])
                self.log("END_RETRIEVAL", detail=False,
                         log_file=str(self.log_path) if self.log_path else None)
            finally:
                if self._file_handler is not None:
                    logger.removeHandler(self._file_handler)
                    self._file_handler.close()
                _active_retrieval_run.reset(self._context_token)


def new_retrieval_run():
    return RetrievalRun(get_author_bio_text, get_recent_post_texts)


def is_disaster_related_text(text):
    return bool(disaster_matches(text))


def find_valid_cebu_localities(text):
    return locality_matches(text or "")


def mentions_cebu_explicitly(text):
    return bool(explicit_matches(text or ""))


def cascade_check_cebu_relevance(post_text, author_did=None, author_handle=None):
    run = new_retrieval_run()
    try:
        decision = run.evaluate(post_text, actor=author_did or author_handle,
                                author_did=author_did, author_handle=author_handle)
        level = {"EXPLICIT_CEBU": "post_text", "AUTHOR_BIO": "author_bio",
                 "AUTHOR_RECENT_POSTS": "author_recent_posts"}.get(decision["verification"])
        return decision["accepted"], level, decision["locations"]
    finally:
        run.finish()


def post_matches_dual_condition(post_text, author_did=None, author_handle=None):
    relevant, _, _ = cascade_check_cebu_relevance(post_text, author_did, author_handle)
    return relevant, bool(disaster_matches(post_text))


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
    with new_retrieval_run() as run:
        run.log("DATE_WINDOW", since=bluesky_since_value, until=until_value,
                incremental=used_incremental_fetch)
        total_fetched = 0
        matched_count = 0
        discarded_count = 0
        seen_uris = set()

        for search_query in location_terms:
            run.query(search_query)
            try:
                search_results = search_bluesky_posts(
                    search_query, max_posts, since=bluesky_since_value, until=until_value
                )
            except RuntimeError as search_error:
                run.error("SEARCH_QUERY", search_error)
                continue

            run.query(search_query, len(search_results))
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
                decision = run.evaluate(post_text, post_id=post_uri,
                                        actor=get_attr(author, "did") or get_attr(author, "handle"),
                                        available_bio=get_attr(author, "description"),
                                        author_did=get_attr(author, "did"),
                                        author_handle=get_attr(author, "handle"), search_query=search_query)
                if not decision["accepted"]:
                    discarded_count += 1
                    continue

                matched_count += 1
                raw_post_document = {
                    **retrieval_metadata(decision),
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

        run.log("PERSISTENCE_SUMMARY", saved=matched_count,
                skipped_already_saved=skipped_already_saved)

        run.finish()
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
    """Discover dialect-only candidates; validate rules before raw persistence."""
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
    with new_retrieval_run() as run:
        run.log("DATE_WINDOW", since=bluesky_since_value, until=until_value,
                incremental=used_incremental_fetch)
        total_fetched = 0
        matched_count = 0
        discarded_count = 0
        seen_uris = set()

        search_terms = sorted({term for term in search_source_terms if len(term) >= 4}
                              | set(location_disaster_queries()))

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
            run.query(search_query)
            try:
                search_results = search_bluesky_posts(
                    search_query, max_posts, since=bluesky_since_value, until=until_value
                )
            except RuntimeError as search_error:
                run.error("SEARCH_QUERY", search_error)
                continue

            run.query(search_query, len(search_results))
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

                decision = run.evaluate(post_text, post_id=post_uri,
                                        actor=author_did or author_handle,
                                        available_bio=get_attr(author, "description"),
                                        author_did=author_did, author_handle=author_handle,
                                        search_query=search_query)
                if not decision["accepted"]:
                    discarded_count += 1
                    continue

                matched_count += 1
                raw_post_document = {
                    **retrieval_metadata(decision),
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

        run.log("PERSISTENCE_SUMMARY", saved=matched_count,
                skipped_already_saved=skipped_already_saved)

        run.finish()
        return {
            "total_fetched": total_fetched,
            "matched": matched_count,
            "discarded": discarded_count,
            "skipped_already_saved": skipped_already_saved,
            "bluesky_since_value": bluesky_since_value,
            "used_incremental_fetch": used_incremental_fetch,
        }
