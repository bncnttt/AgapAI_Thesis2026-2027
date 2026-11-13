import logging
import json
import os
from contextvars import ContextVar
from datetime import datetime, timezone, timedelta
from functools import wraps
from pathlib import Path
from uuid import uuid4
import re
from time import perf_counter
from typing import Any, Dict, List, Optional, Set

import networkx as nx
from fastapi import APIRouter, HTTPException

from agapai_api.clients import (
    authenticate_bluesky,
    client,
    is_bluesky_authenticated,
    mongo_connected,
    posts_col,
    users_col,
)

logger = logging.getLogger("agapai.sna")
router = APIRouter(prefix="/api/sna", tags=["sna"])

_sna_run = ContextVar("sna_log_run", default=None)


class _SnaRunFilter(logging.Filter):
    def __init__(self, run_id=None):
        super().__init__()
        self.run_id = run_id

    def filter(self, record):
        active = _sna_run.get()
        record.sna_run = active or "outside-analysis"
        return self.run_id is None or active == self.run_id


_sna_formatter = logging.Formatter("%(asctime)s [SNA] run=%(sna_run)s %(levelname)s %(message)s")
if not logger.handlers:
    _console = logging.StreamHandler()
    _console.addFilter(_SnaRunFilter())
    _console.setFormatter(_sna_formatter)
    logger.addHandler(_console)
logger.setLevel(logging.INFO)
logger.propagate = False


def log_sna_analysis(operation):
    """Capture each request independently, including simultaneous analyses."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            run_id = uuid4().hex[:10]
            token = _sna_run.set(run_id)
            handler = None
            started = perf_counter()
            try:
                try:
                    directory = Path(os.getenv("AGAPAI_SNA_LOG_DIR") or
                                     Path(__file__).resolve().parents[1] / "logs" / "sna")
                    directory.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
                    log_path = directory / f"sna_{operation}_{timestamp}_{run_id}.log"
                    handler = logging.FileHandler(log_path, encoding="utf-8")
                    handler.addFilter(_SnaRunFilter(run_id))
                    handler.setFormatter(_sna_formatter)
                    logger.addHandler(handler)
                    logger.info("Log file: %s", log_path.resolve())
                except OSError as error:
                    logger.warning("Could not open SNA log file: %s", type(error).__name__)
                logger.info("START %s %s", operation, json.dumps(kwargs, ensure_ascii=False, default=str))
                result = function(*args, **kwargs)
                summary = {key: result[key] for key in ("target", "matches", "jaccard", "graph_stats", "counts")
                           if key in result}
                logger.info("RESULT\n%s", json.dumps(summary, ensure_ascii=False, indent=2, default=str))
                return result
            except Exception as error:
                logger.error("ANALYSIS FAILED: %s", type(error).__name__)
                raise
            finally:
                logger.info("END %s elapsed_seconds=%.3f", operation, perf_counter() - started)
                if handler is not None:
                    logger.removeHandler(handler)
                    handler.close()
                _sna_run.reset(token)
        return wrapped
    return decorate

URGENCY_RANK = {"Very High": 0, "High": 1, "Medium": 2, "Low": 3}

# SNA / graph settings
MAX_HOPS = 3
DEFAULT_HOP_VIEW = 1
LIVE_PAGE_LIMIT = 100
MAX_LIVE_RELATIONSHIPS = 500
MAX_LIVE_EXPANSIONS = 40
NEIGHBOR_SAMPLE_LIMIT = 500

ID_KEYS = ("did", "_id", "id", "handle")
NEED_KEYS = ("needed_items", "needs", "requested_items")
OFFER_KEYS = ("offered_items", "offers", "available_items")
GENERAL_RELIEF_ITEM = "general relief"
NO_POST_TEXT = "No post text available."
DID_PATTERN = re.compile(r"^did:[a-z0-9]+:[A-Za-z0-9._:%-]+$")

def as_string(value: Any) -> str:
    """Return a stable string identifier from dicts, SDK models, or plain values."""
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ID_KEYS:
            if value.get(key):
                return str(value[key])
    for key in ID_KEYS:
        attr = getattr(value, key, None)
        if attr:
            return str(attr)
    return str(value)

def as_string_set(values: Any) -> Set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {s for s in map(as_string, values) if s}


def is_bluesky_did(value: Any) -> bool:
    """Social graph nodes are actor DIDs, never handles, URLs, or AT URIs."""
    return bool(DID_PATTERN.fullmatch(as_string(value).strip()))


def as_did_set(values: Any) -> Set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {
        value
        for value in (as_string(item).strip() for item in values)
        if is_bluesky_did(value)
    }

def normalize_items(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, dict):
        values = value.keys()
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value).replace(";", ",").split(",")
    return [str(item).strip().lower() for item in values if str(item).strip()]

def display_items(items: List[str]) -> List[str]:
    return [item.title() for item in items]

def normalize_urgency(value: Any) -> str:
    text = str(value or "").strip().title()
    return text if text in URGENCY_RANK else "Low"

def normalize_role(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    return {"victim": "Victim", "volunteer": "Volunteer"}.get(text)

def get_post_location(post: Dict[str, Any]) -> str:
    location = post.get("location") or post.get("location_name") or ""
    if isinstance(location, dict):
        return str(location.get("name") or location.get("location_name") or "")
    return str(location)

_analysis_cache: Dict[str, Dict[str, Any]] = {}
_social_cache: Dict[str, Dict[str, Set[str]]] = {}


def stored_semantic_result(post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read semantic fields already persisted during ingestion."""
    result = post.get("classifier_prediction") or post.get("classification")
    if not isinstance(result, dict):
        result = {}

    if result:

        return {
            "role": normalize_role(result.get("role")),
            "intent": str(result.get("intent") or "").lower(),
            "needs": normalize_items(result.get("needs")),
            "offers": normalize_items(result.get("offers")),
            "urgency": result.get("urgency"),
        }
    role = normalize_role(post.get("role"))
    intent = str(result.get("intent") or post.get("intent") or "").lower()
    needs = normalize_items(
        result.get("needs")
        or post.get("needed_items")
        or post.get("needs")
    )
    offers = normalize_items(
        result.get("offers")
        or post.get("offers")
        or post.get("offered_items")
        or post.get("donations")
    )

    if not intent:
        return None

    return {
        "role": role,
        "intent": intent,
        "needs": needs,
        "offers": offers,
        "urgency": result.get("urgency") or post.get("urgency"),
    }

def analyze_post_text(post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = str(post.get("text") or "").strip()
    if not text:
        return None
    if text in _analysis_cache:
        return _analysis_cache[text]
    try:
        from agapai_api.classify import classify

        result = classify(text)
    except Exception:
        logger.exception("SNA could not classify a post")
        return None
    _analysis_cache[text] = result
    return result

def warm_analysis_cache(posts: List[Dict[str, Any]]) -> None:
    texts = [
        t
        for t in dict.fromkeys(str(p.get("text") or "").strip() for p in posts
                               if stored_semantic_result(p) is None)
        if t and t not in _analysis_cache
    ]
    if not texts:
        return
    try:
        from agapai_api.classify import get_classifier

        for offset in range(0, len(texts), 16):
            batch = texts[offset:offset + 16]
            for text, result in zip(batch, get_classifier().predict(batch)):
                _analysis_cache[text] = result
    except Exception:
        logger.exception("SNA could not prepare classifier results")

def _stored_items(post: Dict[str, Any], keys) -> List[str]:
    """Stored metadata: only a fallback when the post text cannot be analyzed."""
    for key in keys:
        items = normalize_items(post.get(key))
        if items:
            return items
    return []

def infer_needed_items(post: Dict[str, Any]) -> List[str]:
    stored = stored_semantic_result(post)
    if stored is not None:
        return stored["needs"]
    analysis = analyze_post_text(post)
    if analysis is not None:
        return normalize_items(list(analysis.get("needs") or {}))
    return []

def infer_offered_items(post: Dict[str, Any]) -> List[str]:
    stored = stored_semantic_result(post)
    if stored is not None:
        return stored["offers"]
    analysis = analyze_post_text(post)
    if analysis is not None:
        return normalize_items(list(analysis.get("offers") or {}))
    return []

def infer_post_role(post: Dict[str, Any]) -> Optional[str]:
    """Use the classifier role independently of its need/offer intent."""
    prediction = stored_semantic_result(post)
    if prediction is None:
        prediction = analyze_post_text(post)
    return normalize_role(prediction.get("role")) if prediction else None


def get_user_document(identifier: str) -> Dict[str, Any]:
    identifier = as_string(identifier)
    if users_col is None or not identifier:
        return {}

    for field in ("_id", "did", "handle"):
        doc = users_col.find_one({field: identifier})
        if doc:
            return doc
    return {}

def resolve_actor_id(identifier: str) -> str:
    """Normalize a route input that may be either a DID or a Bluesky handle."""
    identifier = as_string(identifier).strip()
    if not identifier:
        return identifier

    doc = get_user_document(identifier)
    if doc:
        return as_string(doc.get("did") or doc.get("_id")) or identifier

    try:
        if not is_bluesky_authenticated():
            authenticate_bluesky()
        if is_bluesky_authenticated():
            resolved = as_string(client.get_profile(actor=identifier))
            if resolved:
                return resolved
    except Exception:
        pass

    return identifier

def add_node(nodes_by_id: Dict[str, Dict[str, Any]], node: Dict[str, Any]) -> None:
    node_id = as_string(node.get("id"))
    if not node_id:
        return

    existing = nodes_by_id.get(node_id)
    if not existing or int(node.get("closeness_score", 0)) > int(
        existing.get("closeness_score", 0)
    ):
        nodes_by_id[node_id] = node

def is_help_seeking_victim(post: Dict[str, Any]) -> bool:
    prediction = stored_semantic_result(post)
    if prediction is None:
        prediction = analyze_post_text(post)
    if not isinstance(prediction, dict):
        return False
    return (
        normalize_role(prediction.get("role")) == "Victim"
        and prediction.get("intent") == "need"
        and bool(normalize_items(list(prediction.get("needs") or {})))
    )


@router.get("/incidents")
@log_sna_analysis("queue")
def get_sna_incidents(start: Optional[str] = None, end: Optional[str] = None, author: Optional[str] = None):
    """Build the victim queue from saved posts and the shared classifier."""
    if not mongo_connected or posts_col is None:
        raise HTTPException(status_code=503, detail="MongoDB is not connected.")
    if bool(start) != bool(end):
        raise HTTPException(status_code=422, detail="Provide both start and end dates, or neither.")

    query: Dict[str, Any] = {}
    if start and end:
        if start > end:
            raise HTTPException(status_code=422, detail="start must not be after end.")
        query["created_at"] = {"$gte": f"{start}T00:00:00Z", "$lte": f"{end}T23:59:59.999Z"}
    if author and author.strip():
        pattern = re.escape(author.strip())
        query["$or"] = [
            {"author_handle": {"$regex": pattern, "$options": "i"}},
            {"posted_by": {"$regex": pattern, "$options": "i"}},
            {"author_did": {"$regex": pattern, "$options": "i"}},
        ]

    total = posts_col.count_documents(query)
    checked = role_victims = intent_needs = needs_found = 0
    incidents: List[Dict[str, Any]] = []
    for post in posts_col.find(query).sort("created_at", -1):
        checked += 1
        prediction = stored_semantic_result(post)
        if prediction is None:
            prediction = analyze_post_text(post)
        if prediction is None:
            continue
        role = normalize_role(prediction.get("role"))
        intent = str(prediction.get("intent") or "").lower()
        needs = normalize_items(prediction.get("needs"))
        if role == "Victim":
            role_victims += 1
        if intent == "need":
            intent_needs += 1
        if needs:
            needs_found += 1
        included = role == "Victim" and intent == "need" and bool(needs)
        handle = as_string(post.get("author_handle") or post.get("posted_by") or post.get("author_did"))
        logger.info("[SNA QUEUE] %s role=%s intent=%s needs=%s %s", handle, role, intent, needs,
                    "INCLUDED" if included else "EXCLUDED: not help-seeking")
        if not included:
            continue
        incidents.append({
            "_id": as_string(post.get("_id")),
            "author_did": as_string(post.get("author_did") or post.get("did")),
            "author_handle": handle,
            "role": "Victim",
            "help_seeking": True,
            "urgency_level": normalize_urgency((prediction or {}).get("urgency") or post.get("urgency")),
            "needed_items": display_items(needs),
            "location_name": get_post_location(post),
            "created_at": post.get("created_at"),
        })
    logger.info("[SNA QUEUE] total=%s checked=%s role_victim=%s intent_need=%s needs=%s final=%s",
                total, checked, role_victims, intent_needs, needs_found, len(incidents))
    return {"incidents": incidents, "counts": {"total_posts": total, "checked": checked,
            "role_victims": role_victims, "intent_needs": intent_needs,
            "needs_found": needs_found, "final_incidents": len(incidents)}}

def is_offering_volunteer(post: Dict[str, Any]) -> bool:
    prediction = stored_semantic_result(post)
    if prediction is None:
        prediction = analyze_post_text(post)
    if not prediction:
        return False
    return (
        normalize_role(prediction.get("role")) == "Volunteer"
        and prediction.get("intent") == "offer"
        and bool(normalize_items(prediction.get("offers")))
    )

def find_target_post(user_identifier: str, post_id: Optional[str] = None) -> Dict[str, Any]:
    if not mongo_connected or posts_col is None:
        raise HTTPException(status_code=503, detail="MongoDB is not connected.")

    identifiers = {as_string(user_identifier)}
    resolved = resolve_actor_id(user_identifier)
    if resolved:
        identifiers.add(resolved)

    author = {"author_did": {"$in": list(identifiers)}}
    if post_id:
        post = posts_col.find_one({**author, "_id": post_id})
        if not post:
            raise HTTPException(status_code=404, detail="Post not found for this user.")
        if not (is_help_seeking_victim(post) or is_offering_volunteer(post)):
            raise HTTPException(
                status_code=422,
                detail="This post is not a request for help or an offer of help.",
            )
        return post

    for post in posts_col.find(author).sort("created_at", -1):
        if is_help_seeking_victim(post) or is_offering_volunteer(post):
            return post
    raise HTTPException(
        status_code=404,
        detail="No help-seeking (Victim) or help-offering (Volunteer) post found for this user.",
    )

def get_item_matched_counterparts(
    target_role: str,
    target_items: List[str],
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """Return semantically valid, resource-compatible users for either direction."""
    if posts_col is None or not target_items:
        return []
    target_set = set(normalize_items(target_items))
    counterpart_role = "Victim" if target_role == "Volunteer" else "Volunteer"
    infer = infer_needed_items if counterpart_role == "Victim" else infer_offered_items
    # Text is the source of truth, so do not pre-filter by possibly stale role/item fields.
    candidate_posts = list(posts_col.find({}).sort("created_at", -1).limit(limit * 20))
    warm_analysis_cache(candidate_posts)
    matches: List[Dict[str, Any]] = []
    seen_dids: Set[str] = set()
    for post in candidate_posts:
        valid_role = (
            is_help_seeking_victim(post)
            if counterpart_role == "Victim"
            else is_offering_volunteer(post)
        )
        if not valid_role:
            continue
        counterpart_items = infer(post)
        counterpart_set = set(counterpart_items)
        overlap = sorted(target_set.intersection(counterpart_set))
        if not overlap and GENERAL_RELIEF_ITEM in counterpart_set and target_set:
            overlap = sorted(target_set)
        elif not overlap and GENERAL_RELIEF_ITEM in target_set and counterpart_set:
            overlap = sorted(counterpart_set)
        if not overlap:
            continue
        counterpart_did = as_string(post.get("author_did") or post.get("did"))
        if not counterpart_did or counterpart_did in seen_dids:
            continue
        seen_dids.add(counterpart_did)
        matches.append({
            "did": counterpart_did,
            "handle": as_string(post.get("author_handle") or post.get("posted_by") or counterpart_did),
            "location": get_post_location(post),
            "role": counterpart_role,
            "matched_items": display_items(overlap),
            "matched_count": len(overlap),
            "target_item_count": len(target_set),
            "coverage_ratio": len(overlap) / len(target_set) if target_set else 0.0,
            "post_id": as_string(post.get("_id")),
            "post_text": post.get("text", NO_POST_TEXT),
            "raw_items": counterpart_items,
            "post": post,
        })
        if len(matches) >= limit:
            break
    return matches

def _response_field(response: Any, attribute: str) -> Any:
    if isinstance(response, dict):
        return response.get(attribute)
    return getattr(response, attribute, None)

def _fetch_live_follows(actor: str, followers: bool) -> tuple[bool, Set[str]]:
    edges: Set[str] = set()
    try:
        if not is_bluesky_authenticated():
            authenticate_bluesky()
        if not is_bluesky_authenticated():
            return False, edges
        cursor: Optional[str] = None
        method = client.get_followers if followers else client.get_follows
        attribute = "followers" if followers else "follows"
        seen_cursors: Set[str] = set()
        while len(edges) < MAX_LIVE_RELATIONSHIPS:
            remaining = MAX_LIVE_RELATIONSHIPS - len(edges)
            response = method(actor=actor, cursor=cursor, limit=min(LIVE_PAGE_LIMIT, remaining))
            entries = _response_field(response, attribute)
            for entry in entries if isinstance(entries, list) else []:
                did = as_string(entry.get("did") if isinstance(entry, dict) else entry).strip()
                if is_bluesky_did(did):
                    edges.add(did)
            next_cursor = _response_field(response, "cursor")
            if not next_cursor or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor
    except Exception as error:
        logger.warning("[SNA SOCIAL] %s fetch failed for %s: %s", "followers" if followers else "following", actor, error)
        return False, edges
    return True, edges

def persist_social_snapshot(
    did: str,
    followers: Set[str],
    following: Set[str],
) -> None:
    if not did:
        return

    followers_list = sorted(followers)
    following_list = sorted(following)
    mutuals_list = sorted(followers & following)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

    try:
        if users_col is not None:
            existing = get_user_document(did)
            handle = as_string(existing.get("handle"))
            if not handle:
                try:
                    profile = client.get_profile(actor=did) if is_bluesky_authenticated() else None
                    handle = as_string(_response_field(profile, "handle"))
                except Exception:
                    handle = ""

            update_fields = {
                "follower_count": len(followers_list),
                "following_count": len(following_list),
                "mutual_tie_count": len(mutuals_list),
                "followers": followers_list,
                "following": following_list,
                "mutual_ties": mutuals_list,
                "fetched_at": fetched_at,
            }
            if handle:
                update_fields["handle"] = handle

            result = users_col.update_one(
                {"_id": did},
                {"$set": update_fields},
                upsert=True,
            )
            logger.info(
                "[SNA SOCIAL] persisted %s: matched=%s modified=%s upserted=%s",
                did, result.matched_count, result.modified_count, result.upserted_id,
            )

        if posts_col is not None:
            posts_col.update_many(
                {"author_did": did},
                {
                    "$set": {
                        "social_graph.follower_count": len(followers_list),
                        "social_graph.following_count": len(following_list),
                        "social_graph.mutual_tie_count": len(mutuals_list),
                        "social_graph.followers": followers_list,
                        "social_graph.following": following_list,
                        "social_graph.mutual_ties": mutuals_list,
                        "social_graph.fetched_at": fetched_at,
                    }
                },
            )
    except Exception as error:
        logger.warning("[SNA] Could not persist social snapshot for %s: %s", did, error)


def get_social_sets(did: str) -> Dict[str, Set[str]]:
    did = as_string(did)
    if not did:
        return {"followers": set(), "following": set(), "mutuals": set()}
    if did in _social_cache:
        return _social_cache[did]

    user_doc = get_user_document(did)
    stored_followers = as_did_set(user_doc.get("followers"))
    stored_following = as_did_set(user_doc.get("following"))

    followers_succeeded, live_followers = _fetch_live_follows(did, True)
    following_succeeded, live_following = _fetch_live_follows(did, False)
    followers = live_followers if followers_succeeded else stored_followers
    following = live_following if following_succeeded else stored_following
    data = {
        "followers": followers,
        "following": following,
        "mutuals": followers & following,
    }

    if followers_succeeded or following_succeeded:
        logger.info(
            "[SNA SOCIAL] Actor=%s DID=%s stored followers=%d live followers=%d success=%s "
            "stored following=%d live following=%d success=%s persisted followers=%d following=%d mutuals=%d",
            user_doc.get("handle") or did, did, len(stored_followers), len(live_followers),
            followers_succeeded, len(stored_following), len(live_following), following_succeeded,
            len(followers), len(following), len(followers & following),
        )
        persist_social_snapshot(did, followers, following)
    _social_cache[did] = data
    return data

def get_all_network_edges(did: str) -> Set[str]:
    social = get_social_sets(did)
    return social["followers"] | social["following"]

def build_social_graph(
    root_did: str,
    max_hops: int = MAX_HOPS,
    priority_nodes: Optional[Set[str]] = None,
) -> nx.Graph:
    """Build one undirected NetworkX graph from stored and live Bluesky relationships."""
    graph = nx.Graph()
    graph.add_node(root_did)
    priority_nodes = set(priority_nodes or set())
    for actor in {root_did, *priority_nodes}:
        graph.add_edges_from((actor, n) for n in get_all_network_edges(actor) if n and n != actor)
    expanded: Set[str] = set()
    expansions = 0
    try:
        while expansions < MAX_LIVE_EXPANSIONS:
            distances = nx.single_source_shortest_path_length(graph, root_did, cutoff=max_hops)
            pending = [
                node for node, distance in distances.items()
                if distance < max_hops and node not in expanded
            ]
            if not pending:
                break
            pending.sort(key=lambda node: (distances.get(node, max_hops + 1), node not in priority_nodes))
            for actor in pending:
                if expansions >= MAX_LIVE_EXPANSIONS:
                    break
                graph.add_edges_from((actor, n) for n in get_all_network_edges(actor) if n and n != actor)
                expanded.add(actor)
                expansions += 1
    except Exception as error:
        logger.warning("NetworkX social expansion failed: %s", error)
    return graph

def jaccard_similarity(graph: nx.Graph, left: str, right: str) -> float:
    """NetworkX Jaccard coefficient: |N(left) ∩ N(right)| / |N(left) ∪ N(right)|."""
    if left not in graph or right not in graph or left == right:
        return 0.0

    try:
        return float(next(nx.jaccard_coefficient(graph, [(left, right)]))[2])
    except (StopIteration, nx.NetworkXError, nx.NodeNotFound):
        return 0.0

def shortest_social_path(
    graph: nx.Graph,
    source: str,
    target: str,
    max_hops: int = MAX_HOPS,
) -> Optional[List[str]]:
    """Return the nearest NetworkX path when the target is within max_hops."""
    if source not in graph or target not in graph:
        return None

    try:
        path = nx.shortest_path(graph, source=source, target=target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    return path if len(path) - 1 <= max_hops else None

def get_latest_post_for_user(did: str) -> Dict[str, Any]:
    if posts_col is None:
        return {}
    return posts_col.find_one({"author_did": did}, sort=[("created_at", -1)]) or {}

def classify_bridge_node(victim_did: str, candidate_did: str) -> Dict[str, str]:
    social = get_social_sets(victim_did)
    if candidate_did in social["mutuals"]:
        return {"role": "Mutual Connection", "color": "#0099cc"}
    return {"role": "1-Way Link", "color": "#81d873"}

def _follows(a: str, b: str) -> bool:
    """True when stored or live Bluesky data says a follows b."""
    a_social = get_social_sets(a)
    if b in a_social["following"]:
        return True
    b_social = get_social_sets(b)
    return a in b_social["followers"]

def classify_bridge_edge(prev_did: str, did: str) -> Dict[str, str]:
    """Mutual Connection only when both accounts really follow each other."""
    if _follows(prev_did, did) and _follows(did, prev_did):
        return {"role": "Mutual Connection", "color": "#0099cc"}
    return {"role": "1-Way Link", "color": "#81d873"}

def score_badge_for(jaccard: float, hop: Optional[int]) -> str:
    if hop is None:
        return "⚪ No Social Path"
    return f"Jaccard {jaccard:.2f} · {hop} Hop" + ("" if hop == 1 else "s")

def _role_color(role: str) -> str:
    return "#ff2a2a" if role == "Victim" else "#00b85c"

def _node_payload(
    root_did: str,
    did: str,
    hop: int,
    target_role: str,
    target_post: Dict[str, Any],
    counterpart_matches: Dict[str, Dict[str, Any]],
    as_connection: bool = False,
    prev_did: Optional[str] = None,
    root_needed_items: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if did == root_did:
        needed = (
            root_needed_items
            if root_needed_items is not None
            else infer_needed_items(target_post)
        )
        offered = infer_offered_items(target_post)
        shown_needed, shown_offered = display_items(needed), display_items(offered)
        return {
            "id": did,
            "did": did,
            "handle": as_string(
                target_post.get("author_handle")
                or target_post.get("posted_by")
                or get_user_document(did).get("handle")
                or did
            ),
            "role": target_role,
            "color": _role_color(target_role),
            "val": 10,
            "closeness_score": 100,
            "hop": 0,
            "post_id": as_string(target_post.get("_id")),
            "post_text": target_post.get("text", NO_POST_TEXT),
            "needed_items": shown_needed,
            "offered_items": shown_offered,
            # Field the dashboard inspector reads: needs for a Victim root,
            # offers for a Volunteer root.
            "items": shown_needed if target_role == "Victim" else shown_offered,
            "location": get_post_location(target_post),
            "score_badge": f"Origin ({target_role})",
            "score_breakdown": (
                "Origin node — social distance is measured relative to this node."
            ),
        }

    candidate = counterpart_matches.get(did)
    user_doc = get_user_document(did)

    role = candidate.get("role") if candidate else None
    post = candidate.get("post") if candidate else get_latest_post_for_user(did)
    if not as_connection:
        role = role or infer_post_role(post)
    else:
        role = None

    if role:
        color = _role_color(role)
    else:
        classification = (
            classify_bridge_edge(prev_did, did)
            if prev_did
            else classify_bridge_node(root_did, did)
        )
        role, color = classification["role"], classification["color"]

    candidate = candidate or {}
    node_needed = display_items(infer_needed_items(post))
    node_offered = display_items(infer_offered_items(post))
    return {
        "id": did,
        "did": did,
        "handle": as_string(
            candidate.get("handle")
            or post.get("author_handle")
            or post.get("posted_by")
            or user_doc.get("handle")
            or did
        ),
        "role": role,
        "color": color,
        "val": 10 if role in {"Victim", "Volunteer"} else 3,
        "closeness_score": max(0, 100 - (hop * 25)),
        "hop": hop,
        "post_id": as_string(candidate.get("post_id") or post.get("_id")),
        "post_text": candidate.get("post_text") or post.get("text") or NO_POST_TEXT,
        "needed_items": node_needed,
        "offered_items": node_offered,
        "items": node_offered if role == "Volunteer" else node_needed,
        "location": get_post_location(post),
    }

def _lookup_handle(did: str) -> str:
    """Resolve a readable handle for a path node that has no stored post/user."""
    try:
        if not is_bluesky_authenticated():
            authenticate_bluesky()
        handle = _response_field(client.get_profile(actor=did), "handle")
        return str(handle) if handle else did
    except Exception:
        return did

def _display_handle(did: str) -> str:
    """Prefer a stored handle; resolve live only when the user is not in MongoDB."""
    user_doc = get_user_document(did)
    handle = as_string(user_doc.get("handle"))
    return handle or _lookup_handle(did)

def _sample_social_ids(values: Set[str], priority: Optional[Set[str]] = None,
                       exclude: Optional[Set[str]] = None,
                       limit: int = NEIGHBOR_SAMPLE_LIMIT) -> List[str]:
    """Return a stable sample, keeping path/bridge accounts first when present."""
    priority = set(priority or set())
    exclude = set(exclude or set())
    candidates = [did for did in values if did and did not in exclude]
    candidates.sort(key=lambda did: (did not in priority, did))
    return candidates[:limit]

def build_anchor_neighborhood(
    anchor_did: str,
    anchor_role: str,
    priority_nodes: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
) -> Dict[str, Any]:

    social = get_social_sets(anchor_did)
    followers = _sample_social_ids(
        social["followers"], priority=priority_nodes, exclude=exclude
    )
    following = _sample_social_ids(
        social["following"], priority=priority_nodes, exclude=exclude
    )

    relation_by_did: Dict[str, Set[str]] = {}
    for did in followers:
        relation_by_did.setdefault(did, set()).add("follower")
    for did in following:
        relation_by_did.setdefault(did, set()).add("following")

    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []

    for did, relations in relation_by_did.items():
        mutual = "follower" in relations and "following" in relations
        post = get_latest_post_for_user(did)
        role = "Mutual Connection" if mutual else "1-Way Link"
        color = "#0099cc" if mutual else "#81d873"
        relation_label = "Mutual" if mutual else (
            "Follower" if "follower" in relations else "Following"
        )
        nodes.append({
            "id": did,
            "did": did,
            "handle": _display_handle(did),
            "role": role,
            "color": color,
            "val": 3,
            "closeness_score": 0,
            "hop": None,
            "post_id": as_string(post.get("_id")),
            "post_text": post.get("text") or NO_POST_TEXT,
            "needed_items": display_items(infer_needed_items(post)),
            "offered_items": display_items(infer_offered_items(post)),
            "items": [],
            "location": get_post_location(post),
            "network_anchor": anchor_did,
            "network_anchor_role": anchor_role,
            "relationship": relation_label,
        })

        if mutual:
            links.append({
                "source": anchor_did,
                "target": did,
                "score": 0,
                "label": "Mutual connection",
                "relationship": "mutual",
                "isBridge": True,
            })
        elif "follower" in relations:
            # This account follows the anchor.
            links.append({
                "source": did,
                "target": anchor_did,
                "score": 0,
                "label": "Follower",
                "relationship": "follower",
                "isBridge": False,
            })
        else:
            # The anchor follows this account.
            links.append({
                "source": anchor_did,
                "target": did,
                "score": 0,
                "label": "Following",
                "relationship": "following",
                "isBridge": False,
            })

    return {
        "anchor_id": anchor_did,
        "anchor_role": anchor_role,
        "followers": followers,
        "following": following,
        "followers_shown": len(followers),
        "following_shown": len(following),
        "followers_total": len(social["followers"]),
        "following_total": len(social["following"]),
        "mutual_total": len(social["mutuals"]),
        "nodes": nodes,
        "links": links,
    }

def build_visible_social_links(
    graph: nx.Graph,
    visible_ids: Set[str],
    existing_links: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Expose extra connections among visible sampled nodes without adding more nodes."""
    existing_pairs = {
        frozenset((as_string(link.get("source")), as_string(link.get("target"))))
        for link in existing_links
    }
    links: List[Dict[str, Any]] = []
    for left, right in graph.edges():
        if left not in visible_ids or right not in visible_ids:
            continue
        pair = frozenset((left, right))
        if pair in existing_pairs:
            continue

        left_follows_right = _follows(left, right)
        right_follows_left = _follows(right, left)
        if left_follows_right and right_follows_left:
            source, target, label, relationship = left, right, "Mutual connection", "mutual"
        elif left_follows_right:
            source, target, label, relationship = left, right, "Following", "following"
        elif right_follows_left:
            source, target, label, relationship = right, left, "Following", "following"
        else:
            source, target, label, relationship = left, right, "Social connection", "social"

        links.append({
            "source": source,
            "target": target,
            "score": 0,
            "label": label,
            "relationship": relationship,
            "isBridge": True,
        })
        existing_pairs.add(pair)
    return links

def build_path_view(
    graph: nx.Graph,
    root_did: str,
    hop: int,
    target_role: str,
    target_post: Dict[str, Any],
    counterpart_matches: Dict[str, Dict[str, Any]],
    ranked_matches: List[Dict[str, Any]],
    root_needed_items: Optional[List[str]] = None,
) -> Dict[str, Any]:

    root_node = _node_payload(
        root_did, root_did, 0, target_role, target_post, counterpart_matches,
        root_needed_items=root_needed_items,
    )
    candidates = [m for m in ranked_matches if m["hop"] == hop]  # already ranked
    if not candidates:
        return {
            "hop": hop,
            "nodes": [root_node],
            "links": [],
            "match": None,
            "message": "No eligible volunteer found"
            if target_role == "Victim"
            else "No eligible victim found",
        }

    best = candidates[0]
    path = best["path"]
    nodes = [root_node]
    for index, did in enumerate(path[1:], start=1):
        is_end = index == len(path) - 1
        node = _node_payload(
            root_did, did, index, target_role, target_post, counterpart_matches,
            as_connection=not is_end,
            prev_did=path[index - 1],
        )
        if is_end:
            node["score_badge"] = best["score_badge"]
            node["score_breakdown"] = (
                f"Matched {best.get('matched_count', len(best['matched_items']))}/"
                f"{best.get('target_item_count', len(best['matched_items']))} resources "
                f"({', '.join(best['matched_items'])}); "
                f"{best['hop']} hop(s) away, Jaccard {best['jaccard']:.2f}."
            )
        if node["handle"] == did:
            node["handle"] = _lookup_handle(did)
        nodes.append(node)

    links = [
        {
            "source": left,
            "target": right,
            "score": 0,
            "label": f"Hop {i + 1}",
            "hop": i + 1,
            "isBridge": True,
        }
        for i, (left, right) in enumerate(zip(path, path[1:]))
    ]
    return {
        "hop": hop,
        "nodes": nodes,
        "links": links,
        "match": {key: best[key] for key in ("did", "handle", "jaccard", "hop", "path")},
        "message": None,
    }

def verify_social_path(path: List[str]) -> bool:

    if not path or len(path) < 2:
        return False
    for left, right in zip(path, path[1:]):
        if not (_follows(left, right) or _follows(right, left)):
            logger.warning(
                "[SNA] Rejected unverified social edge: %s <-> %s",
                left,
                right,
            )
            return False
    return True

def evaluate_counterpart(
    graph: nx.Graph,
    root_did: str,
    counterpart: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    candidate_did = counterpart["did"]
    if not candidate_did or candidate_did == root_did:
        return None

    path = shortest_social_path(graph, root_did, candidate_did, max_hops=MAX_HOPS)
    if not path or not verify_social_path(path):
        return None

    hop = len(path) - 1
    jaccard = jaccard_similarity(graph, root_did, candidate_did)

    return {
        "did": candidate_did,
        "handle": counterpart["handle"],
        "role": counterpart["role"],
        "matched_items": counterpart["matched_items"],
        "matched_count": int(counterpart.get("matched_count", len(counterpart.get("matched_items", [])))),
        "target_item_count": int(counterpart.get("target_item_count", 0)),
        "coverage_ratio": float(counterpart.get("coverage_ratio", 0.0)),
        "hop": hop,
        "path": path,
        "jaccard": jaccard,
        "score": round(jaccard * 100),
        "score_badge": score_badge_for(jaccard, hop),
        "post_id": counterpart["post_id"],
        "post_text": counterpart["post_text"],
    }

def rank_counterparts(evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Resource coverage is primary.

    For two or more target needs/offers:
      - prefer candidates covering at least 2 items;
      - allow a 1-item candidate only when no socially-connected candidate
        covers 2 or more items.

    Ranking inside the chosen tier:
      1. more matched resources
      2. lower hop count
      3. higher Jaccard similarity
    """
    if not evaluations:
        return []

    target_item_count = max(
        int(item.get("target_item_count", 0)) for item in evaluations
    )
    preferred_min = 2 if target_item_count >= 2 else 1

    preferred = [
        item for item in evaluations
        if int(item.get("matched_count", 0)) >= preferred_min
    ]
    eligible = preferred or [
        item for item in evaluations
        if int(item.get("matched_count", 0)) >= 1
    ]

    return sorted(
        eligible,
        key=lambda item: (
            -int(item.get("matched_count", 0)),
            int(item.get("hop", MAX_HOPS + 1)),
            -float(item.get("jaccard", 0.0)),
        ),
    )

@router.get("/graph/{user_did}")
@log_sna_analysis("graph")
def get_sna_graph(
    user_did: str, hop: int = DEFAULT_HOP_VIEW, post_id: Optional[str] = None
):
    request_started = perf_counter()
    if hop not in (1, 2, 3):
        raise HTTPException(status_code=422, detail="hop must be 1, 2, or 3.")

    _social_cache.clear()  # pick up newly created Bluesky follows on each dashboard request
    requested_identifier = as_string(user_did)
    target_post = find_target_post(requested_identifier, post_id)

    root_did = as_string(
        target_post.get("author_did")
        or resolve_actor_id(requested_identifier)
        or requested_identifier
    )
    target_role = infer_post_role(target_post) or "Victim"

    needed_items = infer_needed_items(target_post)
    offered_items = infer_offered_items(target_post)

    root_handle = as_string(
        target_post.get("author_handle")
        or target_post.get("posted_by")
        or get_user_document(root_did).get("handle")
        or requested_identifier
    )

    # 1. Find semantically valid, resource-compatible counterparts from post text.
    target_items = needed_items if target_role == "Victim" else offered_items
    counterparts = get_item_matched_counterparts(target_role, target_items) if target_items else []
    counterpart_lookup = {item["did"]: item for item in counterparts if item.get("did")}

    # 2. Build one NetworkX graph and prioritize those counterpart accounts during live hydration.
    graph_started = perf_counter()
    graph = build_social_graph(root_did, max_hops=MAX_HOPS, priority_nodes=set(counterpart_lookup))
    logger.info("[SNA] Build bounded NetworkX graph: %.2fs", perf_counter() - graph_started)

    # 3. Evaluate every resource-compatible candidate independently.
    evaluations = [
        result
        for result in (
            evaluate_counterpart(graph, root_did, counterpart)
            for counterpart in counterparts
        )
        if result
    ]
    ranked_matches = rank_counterparts(evaluations)
    best_match = ranked_matches[0] if ranked_matches else None

    root_social = get_social_sets(root_did)
    root_followers = root_social["followers"]
    root_following = root_social["following"]
    sna_debug = {
        "selected_victim": root_handle,
        "post": target_post.get("text"),
        "semantic_role": target_role,
        "extracted_needs": display_items(needed_items),
        "extracted_offers": display_items(offered_items),
        "followers_count": len(root_followers),
        "following_count": len(root_following),
        "mutual_count": len(root_followers & root_following),
        "graph_type": type(graph).__name__,
        "networkx_nodes": graph.number_of_nodes(),
        "networkx_edges": graph.number_of_edges(),
        "root_neighbors": sorted(graph.neighbors(root_did)) if root_did in graph else [],
        "volunteer_candidates": len(counterparts),
        "resource_compatible_candidates": [c["handle"] for c in counterparts],
        "candidates_by_hop": {
            str(level): [m["handle"] for m in ranked_matches if m["hop"] == level]
            for level in (1, 2, 3)
        },
        "selected_volunteer": best_match["handle"] if best_match else None,
        "selected_path": best_match["path"] if best_match else None,
        "jaccard": best_match["jaccard"] if best_match else None,
    }
    logger.info("SNA debug: %s", sna_debug)
    logger.info("[SNA] Selected-user request completed in %.2fs", perf_counter() - request_started)

    # 4. Build all three dashboard views from the same NetworkX graph.
    hop_views = {
        str(level): build_path_view(
            graph, root_did, level, target_role, target_post, counterpart_lookup,
            ranked_matches, root_needed_items=needed_items,
        )
        for level in (1, 2, 3)
    }
    selected_view = hop_views[str(hop)]

    # 5. Prepare click-to-expand local neighborhoods for the selected Victim
    # and the selected best matching counterpart. Each anchor exposes all retrieved followers/following up to the live safety cap,
    # while the full graph remains available internally for matching/Jaccard.
    neighborhoods: Dict[str, Dict[str, Any]] = {}
    neighborhood_cross_links: List[Dict[str, Any]] = []

    path_priority = set(best_match.get("path", [])[1:-1]) if best_match else set()
    counterpart_did = best_match["did"] if best_match else ""
    root_neighborhood = build_anchor_neighborhood(
        root_did,
        target_role,
        priority_nodes=path_priority,
        exclude={counterpart_did} if counterpart_did else set(),
    )
    neighborhoods[root_did] = root_neighborhood

    if best_match:
        counterpart_role = counterpart_lookup.get(counterpart_did, {}).get(
            "role", "Volunteer" if target_role == "Victim" else "Victim"
        )
        counterpart_neighborhood = build_anchor_neighborhood(
            counterpart_did,
            counterpart_role,
            priority_nodes=path_priority,
            exclude={root_did},
        )
        neighborhoods[counterpart_did] = counterpart_neighborhood

        best_view = hop_views.get(str(best_match["hop"]), selected_view)
        visible_ids = {
            root_did,
            counterpart_did,
            *root_neighborhood["followers"],
            *root_neighborhood["following"],
            *counterpart_neighborhood["followers"],
            *counterpart_neighborhood["following"],
            *best_match.get("path", []),
        }
        base_links = (
            root_neighborhood["links"]
            + counterpart_neighborhood["links"]
            + best_view["links"]
        )
        neighborhood_cross_links = build_visible_social_links(
            graph, visible_ids, base_links
        )

    return {
        "target": {
            "did": root_did,
            "handle": root_handle,
            "role": target_role,
            "urgency_level": normalize_urgency(
                (stored_semantic_result(target_post) or analyze_post_text(target_post) or {}).get("urgency")
            ),
            "needed_items": display_items(needed_items),
            "offered_items": display_items(offered_items),
            "location": get_post_location(target_post),
            "post_id": as_string(target_post.get("_id")),
            "post_text": target_post.get("text", NO_POST_TEXT),
        },
        "view": {
            "type": "jaccard_1_hop" if hop == 1 else f"{hop}_hops",
            "hop": hop,
            "label": "Jaccard / 1 Hop" if hop == 1 else f"{hop} Hops",
        },
        "jaccard": {
            "formula": "|N(A) ∩ N(B)| / |N(A) ∪ N(B)|",
            "algorithm": "NetworkX jaccard_coefficient on the undirected social graph",
            "best_match": (
                {key: best_match[key] for key in ("did", "handle", "jaccard", "hop", "path")}
                if best_match
                else None
            ),
        },
        "debug": sna_debug,
        "matches": ranked_matches,
        "hop_views": hop_views,
        "neighborhood_limit": NEIGHBOR_SAMPLE_LIMIT,
        "neighborhoods": neighborhoods,
        "neighborhood_cross_links": neighborhood_cross_links,
        # Keep the existing top-level shape so the current frontend can render
        # the selected view without a second endpoint.
        "nodes": selected_view["nodes"],
        "links": selected_view["links"],
        "message": selected_view["message"],
        "graph_stats": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "max_supported_hops": MAX_HOPS,
        },
    }
