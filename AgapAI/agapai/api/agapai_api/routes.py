from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
import pymongo

from agapai_api.bluesky import (
    hydrate_social_graph,
    collect_graph_members_with_fallback,
    get_attr,
    resolve_incremental_since,
    search_bluesky_posts,
)
from agapai_api.corpus import get_dialect_only_disaster_terms
from agapai_api.clients import (
    authenticate_bluesky,
    client, 
    db,
    get_bluesky_auth_error,
    is_bluesky_authenticated,
    mongo_connected,
    posts_col,
    users_col,
)
from agapai_api.classify import classify, get_classifier
from agapai_api.geography import (
    extract_location_name,
    get_cebu_location_search_terms,
)
from agapai_api.pipeline import DISASTER_TYPES, evaluate_post
from agapai_api.retrieval import (
    RETRIEVAL_RULE_VERSION,
    location_disaster_queries,
    new_retrieval_run,
    retrieval_metadata,
    retrieve_and_cascade_filter_disaster_posts,
    retrieve_and_filter_cebu_disaster_posts,
)

router = APIRouter()
get_classifier()  # load agapai_multitask_xlmr once at startup, not on the first request
# Bump this whenever classify_disaster.py's LOGIC changes (not just its
# weights) -- enrich_post_with_classifier_fields() below skips
# re-classifying a saved post whenever its stored "classifier_model"
# already equals this string, so an unbumped version silently keeps
# serving stale role/urgency values (e.g. the old "Not Applicable"
# Category-gate output) forever, even after the code is fixed.
CLASSIFIER_MODEL_NAME = "agapai_multitask_xlmr_v1"
CLASSIFIER_FIELD_NAMES = (
    "role",
    "role_confidence",
    "urgency",
    "urgency_confidence",
    "needs",
    "donations",
)

def classify_role_urgency(text, is_disaster_related=True):
    prediction = classify(text)
    prediction.pop("text", None)
    # The model's own need/offer split replaces the old exemplar-based
    # extractor; stored as lists of category names for the dashboard.
    needs = list(prediction["needs"]) if is_disaster_related else []
    donations = list(prediction["offers"]) if is_disaster_related else []
    return {
        "classifier_prediction": prediction,
        "classifier_model": CLASSIFIER_MODEL_NAME,
        "role": prediction["role"],
        "role_confidence": prediction["role_confidence"],
        "urgency": prediction["urgency"],
        "urgency_confidence": prediction["urgency_confidence"],
        "needs": needs,
        "donations": donations,
    }




def enrich_post_with_classifier_fields(document):
    # Posts saved by an older model (or without NEEDS/DONATIONS) are
    # reclassified once with the current model and the result is persisted.
    if document.get("classifier_model") == CLASSIFIER_MODEL_NAME and all(
        document.get(field) is not None for field in CLASSIFIER_FIELD_NAMES
    ):
        return document

    text = document.get("text") or ""
    if not text:
        return document

    try:
        classifier_result = classify_role_urgency(
            text, is_disaster_related=bool(document.get("is_disaster_related"))
        )
    except Exception as classifier_error:
        print(
            "[AGAPAI LOG] WARNING: Saved-post classifier enrichment failed "
            f"for {document.get('_id')}: {classifier_error}"
        )
        return document

    document.update(classifier_result)
    posts_col.update_one(
        {"_id": document["_id"]},
        {"$set": classifier_result},
    )
    return document


def parse_bluesky_datetime(value):
    if not value:
        return None

    clean_value = value.replace("Z", "+00:00")
    if "." in clean_value:
        base_part, nano_part = clean_value.split(".", 1)
        timezone_part = "+00:00" if "+" in nano_part or "-" in nano_part else ""
        clean_value = f"{base_part}.{nano_part[:3]}{timezone_part}"

    parsed = datetime.fromisoformat(clean_value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_date_filter(value, end_of_day=False):
    supported_formats = (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%B %d %Y",
        "%b %d %Y",
    )

    cleaned_value = value.strip()
    parsed = None

    try:
        parsed = datetime.fromisoformat(cleaned_value)
    except ValueError:
        for date_format in supported_formats:
            try:
                parsed = datetime.strptime(cleaned_value, date_format)
                break
            except ValueError:
                continue

    if parsed is None:
        raise HTTPException(
            status_code=422,
            detail="Invalid date format. Use YYYY-MM-DD, MM/DD/YYYY, or Month DD YYYY.",
        )

    if parsed.tzinfo is None:
        if end_of_day:
            parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999000)
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def trim_list(value, graph_limit):
    if not isinstance(value, list):
        return value
    if graph_limit < 0:
        return value
    return value[:graph_limit]


def apply_graph_limit(document, graph_limit):
    if graph_limit < 0:
        return document

    for key in ("followers", "following", "mutual_ties"):
        if key in document:
            document[key] = trim_list(document[key], graph_limit)

    social_graph = document.get("social_graph")
    if isinstance(social_graph, dict):
        for key in ("followers", "following", "mutual_ties"):
            if key in social_graph:
                social_graph[key] = trim_list(social_graph[key], graph_limit)

    return document


def is_saved_disaster_post(document, run=None):
    """Read retrieval acceptance independently of stored model predictions."""
    if (document.get("retrieval_status") == "accepted"
            and document.get("retrieval_rule_version") == RETRIEVAL_RULE_VERSION):
        return True
    own_run = run is None
    run = run or new_retrieval_run()
    try:
        return run.evaluate(document.get("text", ""), post_id=str(document.get("_id", "")),
                            actor=document.get("author_did") or document.get("author_handle"),
                            author_did=document.get("author_did"), author_handle=document.get("author_handle"),
                            search_query=document.get("search_query"))["accepted"]
    finally:
        if own_run:
            run.finish()



def process_and_store_post(raw_post):
    post_dict = raw_post.dict() if hasattr(raw_post, "dict") else dict(raw_post)
    text = get_attr(post_dict, "text", "") or get_attr(
        get_attr(post_dict, "record", {}), "text", ""
    )
    run = new_retrieval_run()
    try:
        author = get_attr(post_dict, "author", {})
        decision = run.evaluate(text, post_id=post_dict.get("uri") or post_dict.get("_id"),
                                actor=post_dict.get("author_did") or get_attr(author, "did"),
                                available_bio=get_attr(author, "description"),
                                author_did=post_dict.get("author_did") or get_attr(author, "did"),
                                author_handle=post_dict.get("author_handle") or get_attr(author, "handle"),
                                search_query=post_dict.get("search_query"))
    finally:
        if "decision" not in locals() or not decision["accepted"]:
            run.finish()
    if not decision["accepted"]:
        return None
    try:
        evaluation = evaluate_post(text)
    except Exception as error:
        run.error("DOWNSTREAM_EVALUATION", error)
        evaluation = {"is_disaster": False, "disaster_type": "none",
                      "disaster_type_confidence": 0.0, "extracted_locations": []}
    try:
        classifier_result = classify_role_urgency(text, is_disaster_related=evaluation["is_disaster"])
    except Exception as error:
        run.error("DOWNSTREAM_CLASSIFIER", error)
        classifier_result = {"classifier_prediction": {}, "role": None, "role_confidence": None,
                             "urgency": None, "urgency_confidence": None, "needs": [], "donations": []}

    post_document = {
        **post_dict,
        "text": text,
        "is_disaster_related": True,
        "disaster_type": next(iter(decision["disasters"])),
        **retrieval_metadata(decision),
        "disaster_type_confidence": evaluation["disaster_type_confidence"],
        "extracted_locations": evaluation["extracted_locations"],
        **classifier_result,
    }

    try:
        return posts_col.insert_one(post_document)
    finally:
        run.finish()

def get_saved_disaster_data(
    posts_query=None,
    users_query=None,
    search_limit=-1,
    graph_limit=10,
    date_filter_applied_since=None,
    date_filter_applied_until=None,
):
    posts_query = posts_query or {}
    users_query = users_query or {}

    post_cursor = posts_col.find(posts_query).sort("created_at", -1)
    user_cursor = users_col.find(users_query).sort("fetched_at", -1)
    if search_limit >= 0:
        post_cursor = post_cursor.limit(search_limit)
        user_cursor = user_cursor.limit(search_limit)

    posts = []
    saved_run = new_retrieval_run()
    try:
        for document in post_cursor:
            if not is_saved_disaster_post(document, saved_run):
                continue
            document = enrich_post_with_classifier_fields(document)
            document["_id"] = str(document["_id"])
            document = apply_graph_limit(document, graph_limit)
            posts.append(document)
    except Exception as error:
        saved_run.error("READ_SAVED_POSTS", error)
        raise
    finally:
        saved_run.finish()
    users = []
    for document in user_cursor:
        document["_id"] = str(document["_id"])
        document = apply_graph_limit(document, graph_limit)
        users.append(document)

    return {
        "status": "success",
        "source": "mongodb",
        "database_preview": {
            "date_filter_applied_since": date_filter_applied_since,
            "date_filter_applied_until": date_filter_applied_until,
            "search_limit": search_limit,
            "graph_limit": graph_limit,
            "posts_returned": len(posts),
            "users_returned": len(users),
        },
        "posts_total": len(posts),
        "users_total": len(users),
        "posts_collection": posts,
        "users_collection": users,
    }


@router.get("/disaster-alerts")
def get_disaster_posts(
    start: Optional[str] = None,
    end: Optional[str] = None,
    search_limit: int = -1,
    graph_limit: int = 10,
    force_refresh: bool = False,
    include_graph: bool = True,
):
    try:
        if not mongo_connected or posts_col is None or users_col is None:
            raise HTTPException(
                status_code=503,
                detail="MongoDB is not connected. Start MongoDB on localhost:27017 and restart the API.",
            )

        if not start and not end and not force_refresh:
            return get_saved_disaster_data(
                search_limit=search_limit,
                graph_limit=graph_limit,
            )

        if (start and not end) or (end and not start):
            raise HTTPException(
                status_code=422,
                detail="Both start and end query parameters are required.",
            )

        if start and end:
            since_dt_utc = parse_date_filter(start)
            until_dt_utc = parse_date_filter(end, end_of_day=True)
        else:
            until_dt_utc = datetime.now(timezone.utc)
            since_dt_utc = until_dt_utc - timedelta(hours=24)

        if since_dt_utc > until_dt_utc:
            raise HTTPException(
                status_code=422,
                detail="start date must be earlier than or equal to end date.",
            )

        since_value = since_dt_utc.isoformat(timespec="seconds").replace("+00:00", "Z")
        until_value = until_dt_utc.isoformat(timespec="seconds").replace("+00:00", "Z")

        if not force_refresh:
            saved_result = get_saved_disaster_data(
                posts_query={
                    "created_at": {
                        "$gte": since_value,
                        "$lte": until_value,
                    }
                },
                users_query={
                    "fetched_at": {
                        "$gte": since_value,
                        "$lte": until_value,
                    }
                },
                search_limit=search_limit,
                graph_limit=graph_limit,
                date_filter_applied_since=since_value,
                date_filter_applied_until=until_value,
            )

            if saved_result["posts_total"] > 0 or saved_result["users_total"] > 0:
                return saved_result

        if not is_bluesky_authenticated():
            authenticate_bluesky()

        if not is_bluesky_authenticated():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Bluesky is not authenticated. Check BLUESKY_HANDLE and "
                    f"BLUESKY_PASSWORD in api/.env. Error: {get_bluesky_auth_error()}"
                ),
            )

        posts_collection = []
        users_collection = []
        seen_users = set()
        seen_posts = set()
        graph_cache = {}
        disaster_texts = []

        inserted_posts_count = 0
        inserted_users_count = 0
        skipped_outside_date_window = 0
        skipped_search_queries = []
        cebu_candidates_checked = 0

        # Complements the location-anchored terms below: a genuine post
        # can name a real Cebu place (e.g. a barangay) without ever saying
        # the word "Cebu" itself, so the anchored terms alone would never
        # surface it as a Bluesky search candidate at all. These bare
        # disaster-term queries widen the candidate pool; the per-post
        # cascade_check_cebu_relevance() call below (not a literal "Cebu"
        # match) is what actually confirms Cebu relevance for them.
        #
        # Only the Tagalog/Cebuano dialect terms are used here, never the
        # English WordNet-expanded ones: an English disaster word (e.g.
        # "blaze", "fire") is also common, unrelated global vocabulary, so
        # searching it bare with no location anchor pulled in large
        # volumes of irrelevant global content (verified case: a UK "New
        # Forest wildfire" post surfaced from a bare "blaze" search).
        run = new_retrieval_run()
        run.log("BUILD_SEARCH_VOCABULARY")
        broad_disaster_terms = sorted({
            term for term in get_dialect_only_disaster_terms() if len(term) >= 4
        })
        cebu_location_terms = sorted(set(get_cebu_location_search_terms()) | set(broad_disaster_terms)
                                     | set(location_disaster_queries()))

        # Dedup against what's already saved: fetched once up front so
        # each candidate post is a cheap in-memory set lookup, instead of
        # a per-post DB round trip. Checked BEFORE has_cebu_context/
        # evaluate_post so re-running "Load Data" doesn't re-run the
        # expensive translation+classification step on posts we already
        # have -- that's the actual cost this dedup needs to avoid.
        existing_post_uris = set(posts_col.distinct("_id"))
        skipped_already_saved = 0

        # True incremental retrieval: ask Bluesky only for what's newer
        # than the latest post we've already saved, instead of re-fetching
        # the full requested range every time. Falls back to the original
        # requested `since_value` when posts_col is empty (first run).
        # since_value/until_value themselves stay untouched above, since
        # they're also used for the saved-data cache-check query, which
        # must still reflect the user's actually requested range.
        bluesky_since_value = resolve_incremental_since(posts_col, fallback_since=since_value)
        used_incremental_fetch = bluesky_since_value != since_value
        run.log("DATE_WINDOW", since=bluesky_since_value, until=until_value,
                incremental=used_incremental_fetch)

        for search_query in cebu_location_terms:
            run.query(search_query)
            try:
                search_results = search_bluesky_posts(
                    search_query,
                    search_limit,
                    since=bluesky_since_value,
                    until=until_value,
                )
            except RuntimeError as search_error:
                run.error("SEARCH_QUERY", search_error)
                skipped_search_queries.append(
                    {
                        "search_query": search_query,
                        "error": str(search_error),
                    }
                )
                continue

            run.query(search_query, len(search_results))
            for post_view in search_results:
                post_uri = get_attr(post_view, "uri")
                if not post_uri or post_uri in seen_posts:
                    continue

                if post_uri in existing_post_uris:
                    skipped_already_saved += 1
                    continue

                seen_posts.add(post_uri)
                record = get_attr(post_view, "record")
                post_text = get_attr(record, "text", "")

                if not record or not post_text:
                    continue

                author = get_attr(post_view, "author")
                author_did = get_attr(author, "did")
                author_handle = get_attr(author, "handle")
                display_name = get_attr(author, "display_name", author_handle)

                # Validate the requested date window before context API calls.
                created = parse_bluesky_datetime(get_attr(record, "created_at"))
                if not created or created < since_dt_utc or created > until_dt_utc:
                    skipped_outside_date_window += 1
                    continue
                decision = run.evaluate(post_text, post_id=post_uri,
                                        actor=author_did or author_handle,
                                        available_bio=get_attr(author, "description"),
                                        author_did=author_did, author_handle=author_handle,
                                        search_query=search_query)
                if not decision["accepted"]:
                    continue
                cebu_matches = decision["locations"]
                cebu_candidates_checked += 1
                try:
                    evaluation = evaluate_post(post_text)
                except Exception as evaluation_error:
                    run.error("DOWNSTREAM_EVALUATION", evaluation_error, post_uri)
                    evaluation = {"is_disaster": False, "disaster_type": "none",
                                  "disaster_type_confidence": 0.0, "extracted_locations": []}

                try:
                    classifier_result = classify_role_urgency(
                        post_text, is_disaster_related=evaluation["is_disaster"]
                    )
                except Exception as classifier_error:
                    run.error("DOWNSTREAM_CLASSIFIER", classifier_error, post_uri)
                    classifier_result = {
                        "classifier_prediction": {},
                        "role": None,
                        "role_confidence": None,
                        "urgency": None,
                        "urgency_confidence": None,
                        "needs": [],
                        "donations": [],
                    }

                seen_posts.add(post_uri)

                created_at_raw = get_attr(record, "created_at")

                try:
                    created_dt_utc = parse_bluesky_datetime(created_at_raw)
                    if (
                        not created_dt_utc
                        or created_dt_utc < since_dt_utc
                        or created_dt_utc > until_dt_utc
                    ):
                        skipped_outside_date_window += 1
                        continue

                    collected_dt_utc = datetime.now(timezone.utc)

                    pht_tz = timezone(timedelta(hours=8))
                    created_dt_local = created_dt_utc.astimezone(pht_tz)
                    collected_dt_local = collected_dt_utc.astimezone(pht_tz)

                    t_created = created_dt_local.strftime(
                        "%A, %B %d, %Y, %I:%M:%S %p PHT"
                    )
                    t_collected = collected_dt_local.strftime(
                        "%A, %B %d, %Y, %I:%M:%S %p PHT"
                    )

                    time_created_readable_value = t_created.replace(", 0", ", ")
                    time_collected_readable_value = t_collected.replace(", 0", ", ")

                    created_at_value = created_dt_utc.isoformat().replace(
                        "+00:00", "Z"
                    )
                    collected_at_value = (
                        collected_dt_utc.isoformat(timespec="milliseconds").replace(
                            "+00:00", "Z"
                        )
                    )
                except Exception:
                    time_created_readable_value = "Unknown Date/Time"
                    time_collected_readable_value = "Unknown Date/Time"
                    created_at_value = created_at_raw
                    collected_at_value = (
                        datetime.now(timezone.utc)
                        .isoformat(timespec="milliseconds")
                        .replace("+00:00", "Z")
                    )

                detected_location = extract_location_name(post_text)
                reply_count = get_attr(post_view, "reply_count", 0)
                repost_count = get_attr(post_view, "repost_count", 0)
                like_count = get_attr(post_view, "like_count", 0)

                official_follower_count = 0
                official_following_count = 0

                if not include_graph:
                    graph_data = {
                        "follower_count": 0,
                        "following_count": 0,
                        "followers": [],
                        "following": [],
                        "mutual_ties": [],
                    }
                elif author_did in graph_cache:
                    graph_data = graph_cache[author_did]
                else:
                    graph_member_limit = (
                        None
                        if graph_limit < 0
                        else max(0, min(graph_limit, 500))
                    )
                    graph_data = hydrate_social_graph(
                        author_did,
                        author_handle,
                        graph_member_limit,
                        posts_col,
                    )
                    profile_display_name = graph_data.pop("display_name", None)
                    if profile_display_name:
                        display_name = profile_display_name
                    graph_cache[author_did] = graph_data

                followers_list = graph_data["followers"]
                following_list = graph_data["following"]
                mutual_ties = graph_data["mutual_ties"]

                post_document = {
                    "_id": post_uri,
                    "author_did": author_did,
                    "author_handle": author_handle,
                    "posted_by": display_name,
                    "text": post_text,
                    "is_disaster_related": True,
                    "disaster_type": next(iter(decision["disasters"])),
                    **retrieval_metadata(decision),
                    "disaster_type_confidence": evaluation["disaster_type_confidence"],
                    **classifier_result,
                    "extracted_locations": evaluation["extracted_locations"],
                    "retrieval_source": "search_posts",
                    "search_query": search_query,
                    "created_at": created_at_value,
                    "time_created_readable": time_created_readable_value,
                    "collected_at": collected_at_value,
                    "time_collected_readable": time_collected_readable_value,
                    "cebu_location_matches": cebu_matches,
                    "reply_count": reply_count,
                    "repost_count": repost_count,
                    "like_count": like_count,
                    "has_location_clue": True if detected_location else False,
                    "location_name": (
                        detected_location
                        if detected_location
                        else "Unspecified Location"
                    ),
                    "processed": False,
                    "social_graph": {
                        "follower_count": graph_data["follower_count"],
                        "following_count": graph_data["following_count"],
                        "followers": followers_list,
                        "following": following_list,
                        "mutual_ties": mutual_ties,
                    },
                }
                posts_collection.append(post_document)
                disaster_texts.append(
                    {
                        "author_did": author_did,
                        "author_handle": author_handle,
                        "posted_by": display_name,
                        "text": post_text,
                        "cebu_location_matches": cebu_matches,
                        "disaster_type": next(iter(decision["disasters"])),
                        "disaster_type_confidence": evaluation["disaster_type_confidence"],
                        "role": classifier_result["role"],
                        "role_confidence": classifier_result["role_confidence"],
                        "urgency": classifier_result["urgency"],
                        "urgency_confidence": classifier_result["urgency_confidence"],
                        "needs": classifier_result["needs"],
                        "donations": classifier_result["donations"],
                        "location_name": (
                            detected_location
                            if detected_location
                            else "Unspecified Location"
                        ),
                        "retrieval_source": "search_posts",
                        "search_query": search_query,
                        "created_at": created_at_value,
                        "time_created_readable": time_created_readable_value,
                    }
                )

                try:
                    posts_col.insert_one(post_document)
                    inserted_posts_count += 1
                except pymongo.errors.DuplicateKeyError:
                    pass

                if author_did not in seen_users:
                    user_document = {
                        "_id": author_did,
                        "handle": author_handle,
                        "display_name": display_name,
                        "follower_count": graph_data["follower_count"],
                        "following_count": graph_data["following_count"],
                        "mutual_tie_count": (
                            len(mutual_ties) if mutual_ties else 0
                        ),
                        "followers": followers_list,
                        "following": following_list,
                        "mutual_ties": mutual_ties,
                        "fetched_at": collected_at_value,
                    }
                    users_collection.append(user_document)
                    seen_users.add(author_did)

                    try:
                        users_col.insert_one(user_document)
                        inserted_users_count += 1
                    except pymongo.errors.DuplicateKeyError:
                        pass

        run.log("PERSISTENCE_SUMMARY", saved_posts=inserted_posts_count,
                saved_users=inserted_users_count, skipped_already_saved=skipped_already_saved,
                skipped_outside_date_window=skipped_outside_date_window)
        run.finish()

        return {
            "status": "success",
            "source": "bluesky_collection",
            "database_preview": {
                "date_filter_applied_since": since_value,
                "date_filter_applied_until": until_value,
                "date_filter_start": start,
                "date_filter_end": end,
                "bluesky_since_value": bluesky_since_value,
                "used_incremental_fetch": used_incremental_fetch,
                "search_limit": search_limit,
                "graph_limit": graph_limit,
                "include_graph": include_graph,
                "cebu_location_terms_used": len(cebu_location_terms),
                "disaster_texts_retrieved": len(disaster_texts),
                "posts_collected_this_cycle": len(posts_collection),
                "users_collected_this_cycle": len(users_collection),
                "newly_saved_to_mongodb_posts": inserted_posts_count,
                "newly_saved_to_mongodb_users": inserted_users_count,
                "skipped_already_saved": skipped_already_saved,
                "skipped_outside_date_window": skipped_outside_date_window,
                "skipped_search_queries": skipped_search_queries,
            },
            "disaster_texts": disaster_texts,
            "posts_collection": posts_collection,
            "users_collection": users_collection,
        }
    except HTTPException:
        if "run" in locals():
            run.finish()
        raise
    except Exception as e:
        if "run" in locals():
            run.error("RETRIEVAL", e)
            run.finish()
        raise HTTPException(
            status_code=400, detail=f"API Processing Error Trace: {str(e)}"
        )


@router.get("/ingest/cebu-posts")
def ingest_cebu_posts(
    start_date: str,
    end_date: str,
    max_posts: int = -1,
    include_barangays: bool = True,
):
    """
    Retrieval trigger: queries Bluesky for posts using dynamically extracted,
    Cebu-anchored PSGC location terms within [start_date, end_date], then
    applies the strict dual-condition filter (Cebu location context AND at
    least one dynamically generated disaster term) before persisting
    anything -- see agapai_api/retrieval.py. Posts that fail either
    condition (e.g. a waterfall photo captioned with just a place name, or
    unrelated crime news) are discarded and never reach MongoDB.
    """
    try:
        if not mongo_connected or db is None:
            raise HTTPException(
                status_code=503,
                detail="MongoDB is not connected. Start MongoDB on localhost:27017 and restart the API.",
            )

        since_dt_utc = parse_date_filter(start_date)
        until_dt_utc = parse_date_filter(end_date, end_of_day=True)
        if since_dt_utc > until_dt_utc:
            raise HTTPException(
                status_code=422,
                detail="start_date must be earlier than or equal to end_date.",
            )

        if not is_bluesky_authenticated():
            authenticate_bluesky()

        if not is_bluesky_authenticated():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Bluesky is not authenticated. Check BLUESKY_HANDLE and "
                    f"BLUESKY_PASSWORD in api/.env. Error: {get_bluesky_auth_error()}"
                ),
            )

        location_terms = get_cebu_location_search_terms(include_barangays=include_barangays)

        anchored_result = retrieve_and_filter_cebu_disaster_posts(
            start_date, end_date, max_posts=max_posts, location_terms=location_terms
        )

        # Complements the location-anchored search above: that search
        # requires the literal word "Cebu" in every query, so it
        # structurally can't find a genuine post that names a real Cebu
        # place (e.g. a barangay) but never says "Cebu" itself. This pass
        # searches disaster terms alone and confirms Cebu relevance via
        # the 3-level cascade (post text -> author bio -> author's recent
        # posts) instead.
        cascade_result = retrieve_and_cascade_filter_disaster_posts(
            start_date, end_date, max_posts=max_posts, include_barangays=include_barangays
        )

        return {
            "status": "success",
            "source": "bluesky_dual_condition_filter_plus_cascade",
            "date_range": {"start_date": start_date, "end_date": end_date},
            "location_terms_used": len(location_terms),
            "total_fetched": anchored_result["total_fetched"] + cascade_result["total_fetched"],
            "raw_posts_saved": anchored_result["matched"] + cascade_result["matched"],
            "discarded": anchored_result["discarded"] + cascade_result["discarded"],
            "skipped_already_saved": (
                anchored_result["skipped_already_saved"] + cascade_result["skipped_already_saved"]
            ),
            "anchored_search": anchored_result,
            "cascade_search": cascade_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"API Processing Error Trace: {str(e)}"
        )
