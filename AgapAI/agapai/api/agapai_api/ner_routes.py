"""NER API: selected posts -> author context -> extraction -> persistence -> events."""
import logging
from datetime import datetime, date, timezone

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agapai_api.clients import posts_col, mongo_connected
from agapai_api.pipeline import DISASTER_TYPES
from agapai_api.ner_algorithm import (
    AUTHOR_HISTORY_LIMIT,
    LOCAL_TIMEZONE,
    NER_ALGORITHM_VERSION,
    get_nlp,
    resolve_post_location,
    extract_explicit_datetime,
    extract_raw_entities,
)
from agapai_api.ner_event_clustering import cluster_posts_into_events, build_event_summary, MAX_TIME_HOURS

router = APIRouter(prefix="/ner", tags=["ner"])

NER_BATCH_SIZE = 100
logger = logging.getLogger(__name__)


class PostSelection(BaseModel):
    post_ids: list[str]


def _selection_query(selection):
    if selection is None:
        return {}
    ids = list(dict.fromkeys(selection.post_ids))
    # Bluesky URIs are strings; older imported records may use BSON IDs.
    return {"_id": {"$in": ids + [ObjectId(value) for value in ids if ObjectId.is_valid(value)]}}


def _scoped_query(scope, query):
    return {"$and": [scope, query]} if scope else query


PROCESSING_PROJECTION = {
    "text": 1,
    "created_at": 1,
    "author_did": 1,
    "author_handle": 1,
    "ner_text_snapshot": 1,
    "ner_created_at_snapshot": 1,
}

EVENT_PROJECTION = {
    "text": 1,
    "author_handle": 1,
    "posted_by": 1,
    "disaster_type": 1,
    "role": 1,
    "ner_locations": 1,
    "created_at": 1,
    "ner_datetime_source": 1,
    "ner_event_datetime": 1,
    "ner_event_date": 1,
    "ner_event_time": 1,
}

TABLE_PROJECTION = {
    "text": 1,
    "role": 1,
    "disaster_type": 1,
    "author_handle": 1,
    "posted_by": 1,
    "ner_locations": 1,
    "ner_ambiguous_locations": 1,
    "ner_datetime": 1,
    "ner_datetime_source": 1,
    "ner_event_datetime": 1,
    "ner_event_date": 1,
    "ner_event_time": 1,
    "created_at": 1,
}


def _needs_ner_processing_query():
    return {
        "$or": [
            {"ner_algorithm_version": {"$ne": NER_ALGORITHM_VERSION}},
            {"ner_processed_at": {"$exists": False}},
            {"ner_text_snapshot": {"$exists": False}},
            {"ner_created_at_snapshot": {"$exists": False}},
            {"$expr": {"$ne": [{"$ifNull": ["$text", None]}, "$ner_text_snapshot"]}},
            {"$expr": {"$ne": [{"$ifNull": ["$created_at", None]}, "$ner_created_at_snapshot"]}},
        ]
    }


def _parse_created_at(value):
    """created_at is stored by the collection pipeline as an ISO-8601 string
    (see routes.py's created_at_value), not a BSON datetime -- parse it here
    rather than assuming it's already a datetime object."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(value, datetime):
        # PyMongo returns naive UTC datetimes by default; Bluesky uses ISO UTC.
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


def _to_ner_datetime(dt_result, created_at):
    if dt_result["source"] in ("explicit", "vague_date"):
        value = dt_result["event_datetime"]
        normalized = value.isoformat() if value else None
        return {
            "source": "extracted_from_text",
            "expressions": [
                {
                    "raw_phrase": dt_result.get("raw_phrase"),
                    "translated_phrase": dt_result.get("translated_phrase"),
                    "temporal_resolution": dt_result.get("temporal_resolution"),
                    "normalized_datetime": normalized,
                }
            ],
        }

    if dt_result["source"] == "fallback":
        fallback_datetime = dt_result["event_datetime"] or created_at
        fallback_readable = fallback_datetime.strftime("%B %d, %Y %I:%M %p") if fallback_datetime else None
        return {"source": "fallback_created_at", "fallback_readable": fallback_readable}

    return {"source": "unresolved"}


def _to_mongo_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return None


def _format_datetime_parts(value):
    if value is None:
        return {"date": None, "time": None}

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        value = value.astimezone(LOCAL_TIMEZONE)
        return {
            "date": value.date().isoformat(),
            "time": value.strftime("%I:%M:%S %p"),
        }

    if isinstance(value, date):
        return {"date": value.isoformat(), "time": None}

    parsed = _parse_created_at(value)
    if parsed:
        return _format_datetime_parts(parsed)

    return {"date": None, "time": None}


def _stored_datetime_component(value):
    if value is None:
        return None
    value = str(value).strip()
    return None if value.casefold() in ("", "none", "null") else value


def _fetch_author_bio(actor):
    from agapai_api.bluesky import get_author_bio_text
    return get_author_bio_text(actor)


def _fetch_previous_post_texts(post):
    """Ten authored posts strictly before this post; exclude reposts and itself."""
    from agapai_api.bluesky import client, get_attr, _call_with_rate_limit_retry

    actor = post.get("author_did") or post.get("author_handle")
    before = _parse_created_at(post.get("created_at"))
    if not actor or before is None:
        return []
    texts, seen, cursors = [], set(), set()
    cursor = None
    while len(texts) < AUTHOR_HISTORY_LIMIT:
        params = {"actor": actor, "limit": NER_BATCH_SIZE}
        if cursor:
            params["cursor"] = cursor
        try:
            response = _call_with_rate_limit_retry(client.app.bsky.feed.get_author_feed, params=params)
        except Exception:
            logger.warning("Could not retrieve author history for NER", exc_info=True)
            break
        for item in get_attr(response, "feed", []) or []:
            candidate = get_attr(item, "post")
            author = get_attr(candidate, "author")
            identity = get_attr(author, "did") if post.get("author_did") else get_attr(author, "handle")
            if identity != actor:
                continue
            uri = get_attr(candidate, "uri")
            record = get_attr(candidate, "record")
            created_at = _parse_created_at(get_attr(record, "created_at"))
            if not uri or uri == post["_id"] or uri in seen or created_at is None or created_at >= before:
                continue
            seen.add(uri)
            # Empty posts also count toward the preceding ten, not ten matches.
            texts.append(get_attr(record, "text", "") or "")
            if len(texts) == AUTHOR_HISTORY_LIMIT:
                break
        cursor = get_attr(response, "cursor")
        if not cursor or cursor in cursors:
            break
        cursors.add(cursor)
    return texts


def _resolve_with_author_context(post, doc, nearby_points, context_cache):
    text = post.get("text") or ""
    kwargs = {"doc": doc, "nearby_high_confidence_points": nearby_points}
    result = resolve_post_location(text, **kwargs)
    actor = post.get("author_did") or post.get("author_handle")
    if result["locations"] or result["ambiguous_locations"]:
        return result
    if not actor:
        return result
    bio_key = ("bio", actor)
    if bio_key not in context_cache:
        context_cache[bio_key] = _fetch_author_bio(actor)
    kwargs["author_bio"] = context_cache[bio_key]
    result = resolve_post_location(text, **kwargs)
    if result["locations"] or result["ambiguous_locations"]:
        return result
    history_key = ("history", actor, str(post["_id"]))
    if history_key not in context_cache:
        context_cache[history_key] = _fetch_previous_post_texts(post)
    return resolve_post_location(
        text, previous_post_texts=context_cache[history_key], **kwargs,
    )


def _process_post(post, nearby_high_confidence_points=None, context_cache=None):
    text = post.get("text") or ""
    created_at = _parse_created_at(post.get("created_at"))
    reference_date = created_at

    context_cache = context_cache if context_cache is not None else {}
    doc_key = ("doc", text)
    if doc_key not in context_cache:
        context_cache[doc_key] = get_nlp()(text)
    doc = context_cache[doc_key]
    location_result = _resolve_with_author_context(
        post, doc, nearby_high_confidence_points,
        context_cache,
    )

    ner_locations = [
        {
            "location_text": loc["location_text"],
            "matched_name": loc["matched_name"],
            "confidence": loc["confidence"],
            "confidence_source": loc.get("confidence_source"),
            "resolution_method": loc.get("resolution_method"),
            "latitude": loc["coordinates"]["lat"],
            "longitude": loc["coordinates"]["lon"],
            "psgc_code": loc.get("psgc_code"),
            "breadcrumb": loc.get("breadcrumb"),
        }
        for loc in location_result["locations"]
    ]
    ner_ambiguous_locations = location_result["ambiguous_locations"]

    time_key = ("time", text, reference_date)
    if time_key not in context_cache:
        context_cache[time_key] = extract_explicit_datetime(text, reference_date=reference_date)
    dt_result = context_cache[time_key]
    ner_datetime = _to_ner_datetime(dt_result, created_at)

    raw_entities = extract_raw_entities(text, doc)

    return {
        "ner_algorithm_version": NER_ALGORITHM_VERSION,
        "ner_locations": ner_locations,
        "ner_ambiguous_locations": ner_ambiguous_locations,
        "ner_datetime": ner_datetime,
        "ner_datetime_source": dt_result["datetime_source"],
        "ner_event_datetime": _to_mongo_datetime(dt_result["event_datetime"]),
        "ner_event_date": dt_result["event_date"],
        "ner_event_time": dt_result["event_time"],
        "raw_locations_found": raw_entities["locations"],
        "ner_persons": raw_entities["persons"],
        "ner_organizations": raw_entities["organizations"],
    }


def _location_rank(location):
    confidence = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(location.get("confidence"), 0)
    specificity = sum(bool(level) for level in location.get("breadcrumb") or [])
    return confidence, specificity


def _direct_anchor_locations(locations):
    direct = [loc for loc in locations if loc.get("confidence") == "HIGH" and loc.get("resolution_method") == "gazetteer"]
    rank = max(map(_location_rank, direct), default=None)
    return [loc for loc in direct if _location_rank(loc) == rank]


def _load_saved_high_confidence_points(excluded_ids, scope=None):
    query = {
        "_id": {"$nin": excluded_ids},
        "ner_processed_at": {"$exists": True},
        "ner_algorithm_version": NER_ALGORITHM_VERSION,
        "ner_text_snapshot": {"$exists": True},
        "ner_created_at_snapshot": {"$exists": True},
        "ner_locations.0": {"$exists": True},
        "created_at": {"$exists": True, "$ne": None},
        "$expr": {
            "$and": [
                {"$eq": [{"$ifNull": ["$text", None]}, "$ner_text_snapshot"]},
                {"$eq": [{"$ifNull": ["$created_at", None]}, "$ner_created_at_snapshot"]},
            ]
        },
    }

    points = []
    for post in posts_col.find(_scoped_query(scope, query), {"ner_locations": 1, "created_at": 1}):
        created_at = _parse_created_at(post.get("created_at"))
        if created_at is None:
            continue

        for loc in _direct_anchor_locations(post.get("ner_locations", [])):
            points.append({
                "post_id": post["_id"],
                "lat": loc["latitude"],
                "lon": loc["longitude"],
                "created_at": created_at,
            })

    return points


def _nearby_points(post, points):
    created_at = _parse_created_at(post.get("created_at"))
    if created_at is None:
        return []
    return [
        {"lat": point["lat"], "lon": point["lon"]}
        for point in points
        if point["post_id"] != post["_id"]
        and abs((point["created_at"] - created_at).total_seconds()) / 3600 <= MAX_TIME_HOURS
    ]


def _refine_selected_locations(scope, context_cache):
    # Revisit earlier batches once all selected anchors are available. Also
    # recompute inferred locations so an earlier selection cannot supply context.
    points = _load_saved_high_confidence_points([], scope)
    query = _scoped_query(scope, {"$or": [
        {"ner_ambiguous_locations.0": {"$exists": True}},
        {"ner_locations.confidence": {"$in": ["MEDIUM", "LOW"]}},
        {"ner_locations.resolution_method": {"$in": ["author_context", "spatiotemporal_centroid"]}},
        {"ner_locations": []},
    ]})
    for post in posts_col.find(query, PROCESSING_PROJECTION):
        fields = _process_post(post, nearby_high_confidence_points=_nearby_points(post, points), context_cache=context_cache)
        posts_col.update_one({"_id": post["_id"]}, {"$set": fields})


@router.post("/process-all")
def process_all_posts(
    limit: int = Query(NER_BATCH_SIZE, ge=1, le=NER_BATCH_SIZE),
    selection: PostSelection | None = None,
):
    if not mongo_connected:
        raise HTTPException(status_code=503, detail="MongoDB is not connected.")

    scope = _selection_query(selection)
    context_cache = {}
    processing_query = _scoped_query(scope, _needs_ner_processing_query())
    all_posts = list(posts_col.find(processing_query, PROCESSING_PROJECTION).sort("_id", 1).limit(limit))
    if not all_posts:
        if selection is not None:
            _refine_selected_locations(scope, context_cache)
        return {"processed": 0, "remaining": 0, "batch_size": limit}

    batch_ids = [post["_id"] for post in all_posts]

    # Pass 1: resolve every post with Rules 1+2 (Rule 3 needs the results of
    # this very pass, so it can't run yet) and collect every HIGH-confidence
    # resolved point with its timestamp along the way.
    fields_by_id = {}
    high_confidence_points = _load_saved_high_confidence_points(batch_ids, scope)
    for post in all_posts:
        fields = _process_post(post, context_cache=context_cache)
        fields_by_id[post["_id"]] = fields

        created_at = _parse_created_at(post.get("created_at"))
        if created_at is None:
            continue
        for loc in _direct_anchor_locations(fields["ner_locations"]):
            high_confidence_points.append({
                "post_id": post["_id"],
                "lat": loc["latitude"],
                "lon": loc["longitude"],
                "created_at": created_at,
            })

    # Pass 2 (Rule 3): any post still carrying ambiguous locations gets one
    # more attempt, now that every HIGH-confidence point in this batch and
    # its timestamp is known -- narrowed to the same 6-hour clustering window
    # used elsewhere (Part B/C), so this stays consistent with the rest of
    # the algorithm's time-window semantics rather than inventing a new one.
    for post in all_posts:
        fields = fields_by_id[post["_id"]]
        if not fields["ner_ambiguous_locations"]:
            continue

        nearby_points = _nearby_points(post, high_confidence_points)
        if not nearby_points:
            continue

        fields_by_id[post["_id"]] = _process_post(post, nearby_high_confidence_points=nearby_points, context_cache=context_cache)

    updated = 0
    for post in all_posts:
        fields = fields_by_id[post["_id"]]
        fields["ner_processed_at"] = datetime.utcnow()
        fields["ner_text_snapshot"] = post.get("text")
        fields["ner_created_at_snapshot"] = post.get("created_at")
        posts_col.update_one({"_id": post["_id"]}, {"$set": fields})
        updated += 1

    remaining = posts_col.count_documents(processing_query)
    if remaining == 0 and selection is not None:
        _refine_selected_locations(scope, context_cache)

    return {"processed": updated, "remaining": remaining, "batch_size": limit}


def _build_event_lookup(scope=None):
    """Group resolved post locations and return post-to-event links plus events."""
    cluster_input = []
    for post in posts_col.find(_scoped_query(scope, {
        "disaster_type": {"$in": DISASTER_TYPES},
        "ner_locations.0": {"$exists": True},
        "created_at": {"$exists": True, "$ne": None},
    }), EVENT_PROJECTION):
        created_at = _parse_created_at(post.get("created_at"))
        if created_at is None:
            continue

        # Treat each resolved place as its own spatial report. A post that
        # mentions several places can therefore appear in more than one event.
        locations_by_key = {}
        for location in post.get("ner_locations", []):
            latitude = location.get("latitude")
            longitude = location.get("longitude")
            if latitude is None or longitude is None:
                continue
            try:
                latitude, longitude = float(latitude), float(longitude)
            except (TypeError, ValueError):
                continue
            key = location.get("psgc_code") or f"{latitude:.6f}:{longitude:.6f}"
            existing = locations_by_key.get(str(key))
            if existing is None or _location_rank(location) > _location_rank(existing):
                locations_by_key[str(key)] = {**location, "latitude": latitude, "longitude": longitude}

        for location_key, location in locations_by_key.items():
            cluster_input.append({
                "_id": post["_id"],
                "text": post.get("text", ""),
                "author_handle": post.get("author_handle"),
                "posted_by": post.get("posted_by"),
                "role": post.get("role"),
                "disaster_type": post["disaster_type"],
                "lat": location["latitude"],
                "lon": location["longitude"],
                "location_key": location_key,
                "location_text": location.get("matched_name"),
                "location_confidence": location.get("confidence"),
                "created_at": created_at,
                "datetime_source": post.get("ner_datetime_source", "unresolved"),
                "event_datetime": _parse_created_at(post.get("ner_event_datetime")),
                "event_date": post.get("ner_event_date"),
                "event_time": post.get("ner_event_time"),
            })

    clusters = cluster_posts_into_events(cluster_input)

    events = []
    post_id_to_event_ids = {}
    for index, cluster in enumerate(clusters):
        summary = build_event_summary(cluster)
        event_id = f"event-{index}"
        event_datetime = summary["event_datetime"]

        posts_payload = []
        seen_post_ids = set()
        for post in cluster:
            post_id = str(post["_id"])
            if post_id in seen_post_ids:
                continue
            seen_post_ids.add(post_id)
            posts_payload.append({
                "_id": post_id,
                "text": post["text"],
                "author_handle": post.get("author_handle"),
                "posted_by": post.get("posted_by"),
                "role": post.get("role"),
            })
            linked_events = post_id_to_event_ids.setdefault(post_id, [])
            if event_id not in linked_events:
                linked_events.append(event_id)

        datetime_parts = _format_datetime_parts(event_datetime)
        # BSON stores instants in UTC. Keep the date/time as extracted from the
        # anchoring post so the map and table agree even across UTC midnight.
        time_anchor = next((
            p for p in cluster
            if p["datetime_source"] in ("explicit", "vague_date")
            and p["event_datetime"] == event_datetime
        ), None)
        if time_anchor is None:
            time_anchor = next((p for p in cluster if p["created_at"] == event_datetime), {})
        if time_anchor:
            anchor_date = _stored_datetime_component(time_anchor.get("event_date"))
            anchor_time = _stored_datetime_component(time_anchor.get("event_time"))
            posted_parts = _format_datetime_parts(time_anchor.get("created_at"))
            datetime_parts["date"] = anchor_date or datetime_parts["date"]
            datetime_parts["time"] = anchor_time or posted_parts["time"]

        events.append({
            "event_id": event_id,
            "lat": summary["location"]["lat"],
            "lon": summary["location"]["lon"],
            "place": summary["location"]["place"],
            "locations": summary["locations"],
            "disaster_type": summary["disaster_type"],
            "event_datetime": event_datetime.isoformat() if event_datetime else None,
            "event_date": datetime_parts["date"],
            "event_time": datetime_parts["time"],
            "post_count": len(posts_payload),
            "posts": posts_payload,
        })

    return events, post_id_to_event_ids


def _table_row(post, post_id_to_event_ids):
    post["_id"] = str(post["_id"])
    post["event_ids"] = post_id_to_event_ids.get(post["_id"], [])
    post["event_id"] = post["event_ids"][0] if post["event_ids"] else None
    event_datetime = _parse_created_at(post.get("ner_event_datetime"))
    event_parts = _format_datetime_parts(event_datetime)
    posted_parts = _format_datetime_parts(_parse_created_at(post.get("created_at")))
    event_date = _stored_datetime_component(post.get("ner_event_date"))
    event_time = _stored_datetime_component(post.get("ner_event_time"))
    post["ner_event_date"] = event_date or (event_parts["date"] if event_datetime else posted_parts["date"])
    if not event_time:
        source = post.get("ner_datetime_source")
        event_time = event_parts["time"] if event_datetime and source in ("explicit", "fallback") else posted_parts["time"]
    post["ner_event_time"] = event_time
    return post


@router.post("/results")
def get_selected_results(selection: PostSelection):
    """Return the loaded posts and their clusters from the same event lookup."""
    if not mongo_connected:
        raise HTTPException(status_code=503, detail="MongoDB is not connected.")
    scope = _selection_query(selection)
    events, event_ids = _build_event_lookup(scope)
    rows = [
        _table_row(post, event_ids)
        for post in posts_col.find(scope, TABLE_PROJECTION).sort("created_at", -1)
    ]
    return {"rows": rows, "events": events, "total": len(rows)}


@router.get("/table-data")
def get_table_data(
    page: int = Query(1, ge=1),
    limit: int = Query(NER_BATCH_SIZE, ge=1, le=NER_BATCH_SIZE),
):
    if not mongo_connected:
        raise HTTPException(status_code=503, detail="MongoDB is not connected.")

    _, post_id_to_event_id = _build_event_lookup()

    skip = (page - 1) * limit
    total = posts_col.count_documents({})
    rows = []
    for post in posts_col.find({}, TABLE_PROJECTION).sort("created_at", -1).skip(skip).limit(limit):
        rows.append(_table_row(post, post_id_to_event_id))

    return {
        "rows": rows,
        "page": page,
        "limit": limit,
        "total": total,
        "has_next": skip + len(rows) < total,
    }


@router.get("/events")
def get_events():
    if not mongo_connected:
        raise HTTPException(status_code=503, detail="MongoDB is not connected.")

    events, _ = _build_event_lookup()
    return {"events": events}
