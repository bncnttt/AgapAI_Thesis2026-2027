import logging
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

URGENCY_RANK = {"High": 0, "Medium": 1, "Low": 2}

# SNA / graph settings
MAX_HOPS = 3
DEFAULT_HOP_VIEW = 1
LIVE_PAGE_LIMIT = 100
MAX_LIVE_GRAPH_NODES = 400

ID_KEYS = ("did", "_id", "id", "handle")
NEED_KEYS = ("needed_items", "needs", "requested_items")
OFFER_KEYS = ("offered_items", "offers", "available_items")
NO_POST_TEXT = "No post text available."


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


def normalize_items(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [
        item.strip().lower()
        for item in str(value).replace(";", ",").split(",")
        if item.strip()
    ]


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


def analyze_post_text(post: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Run the project's multitask XLM-R classifier (intent need/offer, role and
    resource categories) on the post text. The text is the source of truth;
    returns None only when there is no text or the model is unavailable.
    """
    text = str(post.get("text") or "").strip()
    if not text:
        return None
    if text in _analysis_cache:
        return _analysis_cache[text]
    try:
        from agapai_api.classify import classify

        result = classify(text)
    except Exception:
        return None
    _analysis_cache[text] = result
    return result


def warm_analysis_cache(posts: List[Dict[str, Any]]) -> None:
    """Classify many posts in one batched model call."""
    texts = [
        t
        for t in dict.fromkeys(str(p.get("text") or "").strip() for p in posts)
        if t and t not in _analysis_cache
    ]
    if not texts:
        return
    try:
        from agapai_api.classify import classify_batch

        for text, result in zip(texts, classify_batch(texts)):
            _analysis_cache[text] = result
    except Exception:
        pass


def _stored_items(post: Dict[str, Any], keys) -> List[str]:
    """Stored metadata: only a fallback when the post text cannot be analyzed."""
    for key in keys:
        items = normalize_items(post.get(key))
        if items:
            return items
    return normalize_items(post.get("search_query"))


def infer_needed_items(post: Dict[str, Any]) -> List[str]:
    analysis = analyze_post_text(post)
    if analysis is not None:
        return normalize_items(list(analysis.get("needs") or {}))
    return _stored_items(post, NEED_KEYS)


def infer_offered_items(post: Dict[str, Any]) -> List[str]:
    analysis = analyze_post_text(post)
    if analysis is not None:
        return normalize_items(list(analysis.get("offers") or {}))
    return _stored_items(post, OFFER_KEYS)


def infer_post_role(post: Dict[str, Any]) -> Optional[str]:
    """
    Role from the post text via the classifier (its need/offer intent decides
    Victim vs Volunteer). Stored role/needs/offers are only a fallback.
    """
    analysis = analyze_post_text(post)
    if analysis is not None:
        intent = analysis.get("intent")
        if intent == "need":
            return "Victim"
        if intent == "offer":
            return "Volunteer"
        role = normalize_role(analysis.get("role"))
        if role:
            return role

    role = normalize_role(post.get("role"))
    if role:
        return role

    needed = normalize_items(post.get("needed_items") or post.get("needs"))
    offered = normalize_items(post.get("offered_items") or post.get("offers"))
    if offered and not needed:
        return "Volunteer"
    if needed and not offered:
        return "Victim"
    return None


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

    # Resolve through the authenticated Bluesky client when available;
    # otherwise keep the original identifier.
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
    """
    A Victim incident must be a semantic request for help: the classifier's
    role is Victim, its intent is "need", and at least one resource need was
    extracted. A disaster observation (urgency/disaster/location only) fails.
    Uses the classification stored at ingestion when present, else the text.
    """
    prediction = post.get("classifier_prediction") or analyze_post_text(post)
    if not prediction:
        return False
    return (
        normalize_role(prediction.get("role")) == "Victim"
        and prediction.get("intent") == "need"
        and bool(prediction.get("needs"))
    )


def is_offering_volunteer(post: Dict[str, Any]) -> bool:
    prediction = post.get("classifier_prediction") or analyze_post_text(post)
    if not prediction:
        return False
    return prediction.get("intent") == "offer" and bool(prediction.get("offers"))


def find_target_post(user_identifier: str, post_id: Optional[str] = None) -> Dict[str, Any]:
    """
    The post the SNA is rooted at. Must be a real help request (Victim) or a
    real offer (Volunteer) -- never just a post that has urgency/disaster/
    location metadata. When post_id is given, that exact post is used.
    """
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
    """
    Resource compatibility is role-aware and bidirectional:

    Victim     -> search Volunteer posts where Volunteer.offers overlaps Victim.needs.
    Volunteer  -> search Victim posts where Victim.needs overlaps Volunteer.offers.
    """
    if posts_col is None or not target_items:
        return []

    target_set = set(normalize_items(target_items))
    is_volunteer = target_role == "Volunteer"

    if is_volunteer:
        counterpart_role, item_field, infer = "Victim", NEED_KEYS, infer_needed_items
    else:
        counterpart_role, item_field, infer = "Volunteer", OFFER_KEYS, infer_offered_items

    query: Dict[str, Any] = {
        "$or": [
            {"role": counterpart_role},
            {"role": counterpart_role.lower()},
            {item_field[0]: {"$exists": True}},
            {item_field[1]: {"$exists": True}},
        ]
    }

    matches: List[Dict[str, Any]] = []
    seen_dids: Set[str] = set()

    candidate_posts = list(posts_col.find(query).sort("created_at", -1).limit(limit * 8))
    warm_analysis_cache(candidate_posts)

    for post in candidate_posts:
        post_role = infer_post_role(post)
        if post_role and post_role != counterpart_role:
            continue

        counterpart_items = infer(post)
        overlap = sorted(target_set.intersection(counterpart_items))
        if not overlap:
            continue

        counterpart_did = as_string(post.get("author_did") or post.get("did"))
        if not counterpart_did or counterpart_did in seen_dids:
            continue
        seen_dids.add(counterpart_did)

        matches.append(
            {
                "did": counterpart_did,
                "handle": as_string(
                    post.get("author_handle") or post.get("posted_by") or counterpart_did
                ),
                "location": get_post_location(post),
                "role": counterpart_role,
                "matched_items": display_items(overlap),
                "post_id": as_string(post.get("_id")),
                "post_text": post.get("text", NO_POST_TEXT),
                "raw_items": counterpart_items,
                "post": post,
            }
        )

        if len(matches) >= limit:
            break

    return matches


def _response_field(response: Any, attribute: str) -> Any:
    if isinstance(response, dict):
        return response.get(attribute)
    return getattr(response, attribute, None)


def _fetch_live_follows(actor: str, followers: bool) -> Set[str]:
    """
    Read public Bluesky social edges for one actor, following pagination until
    the service stops returning a cursor.
    """
    edges: Set[str] = set()

    try:
        if not is_bluesky_authenticated():
            authenticate_bluesky()
        if not is_bluesky_authenticated():
            return edges

        cursor: Optional[str] = None
        method = client.get_followers if followers else client.get_follows
        attribute = "followers" if followers else "follows"
        seen_cursors: Set[str] = set()

        while True:
            response = method(actor=actor, cursor=cursor, limit=LIVE_PAGE_LIMIT)

            entries = _response_field(response, attribute)
            for entry in entries if isinstance(entries, list) else []:
                did = as_string(entry.get("did") if isinstance(entry, dict) else entry)
                if did:
                    edges.add(did)

            next_cursor = _response_field(response, "cursor")
            if not next_cursor or next_cursor in seen_cursors:
                break

            seen_cursors.add(next_cursor)
            cursor = next_cursor

            # Protect the API from unexpectedly huge social graphs.
            if len(edges) >= LIVE_PAGE_LIMIT:
                break

    except Exception:
        # The MongoDB graph remains a valid fallback when Bluesky is unavailable.
        pass

    return edges


def get_all_network_edges(did: str) -> Set[str]:
    """Return social neighbors from stored data, supplemented by live Bluesky data."""
    user_doc = get_user_document(did)
    stored = as_string_set(user_doc.get("followers")) | as_string_set(
        user_doc.get("following")
    )
    return (
        stored
        | _fetch_live_follows(did, followers=True)
        | _fetch_live_follows(did, followers=False)
    )


def build_social_graph(root_did: str, max_hops: int = MAX_HOPS) -> nx.Graph:
    """
    Build the undirected NetworkX social graph used by the SNA (follower and
    following relationships are treated as adjacency for Jaccard and hops).

    MongoDB relationships are loaded first, then supplemented from the public
    Bluesky follow/follower endpoints.
    """
    graph = nx.Graph()
    graph.add_node(root_did)

    # Seed graph from all locally stored users first.
    if users_col is not None:
        projection = {"_id": 1, "did": 1, "handle": 1, "followers": 1, "following": 1}

        for user_doc in users_col.find({}, projection):
            actor = as_string(user_doc.get("did") or user_doc.get("_id"))
            if not actor:
                continue

            graph.add_node(actor)
            neighbors = as_string_set(user_doc.get("followers")) | as_string_set(
                user_doc.get("following")
            )
            graph.add_edges_from((actor, n) for n in neighbors if n and n != actor)

    # Make sure the selected root's live social connections are present.
    graph.add_edges_from(
        (root_did, n) for n in get_all_network_edges(root_did) if n != root_did
    )

    # Expand only as far as the requested hop depth (graph hydration, not
    # scoring), capped so very large real-world accounts cannot blow up.
    if max_hops >= 2 and graph.number_of_nodes() < MAX_LIVE_GRAPH_NODES:
        try:
            expanded: Set[str] = set()

            # Recompute the neighborhood after each expansion so live edges
            # can populate every layer up to max_hops.
            for _level in range(max_hops):
                distances = nx.single_source_shortest_path_length(
                    graph, root_did, cutoff=max_hops
                )
                current_level = [
                    node
                    for node, distance in distances.items()
                    if distance < max_hops and node not in expanded
                ]
                if not current_level:
                    break

                for actor in current_level:
                    if graph.number_of_nodes() >= MAX_LIVE_GRAPH_NODES:
                        break
                    graph.add_edges_from(
                        (actor, n)
                        for n in get_all_network_edges(actor)
                        if n and n != actor
                    )
                    expanded.add(actor)

                if graph.number_of_nodes() >= MAX_LIVE_GRAPH_NODES:
                    break

        except Exception:
            # Keep the locally constructed graph if NetworkX expansion fails.
            pass

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
    """
    - Mutual Connection = present in both root followers and following.
    - 1-Way Link = present in only one of them.
    """
    victim_doc = get_user_document(victim_did)
    followers = as_string_set(victim_doc.get("followers"))
    following = as_string_set(victim_doc.get("following"))

    if candidate_did in followers and candidate_did in following:
        return {"role": "Mutual Connection", "color": "#0099cc"}
    return {"role": "1-Way Link", "color": "#81d873"}


def _follows(a: str, b: str) -> bool:
    """True when stored data says `a` follows `b` (a.following or b.followers)."""
    a_doc, b_doc = get_user_document(a), get_user_document(b)
    return b in as_string_set(a_doc.get("following")) or a in as_string_set(
        b_doc.get("followers")
    )


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
        role = None  # intermediaries are never classified by their own post role

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
    """
    Display subgraph for one hop level: the root plus the single best match at
    exactly `hop` hops and only the intermediaries on its path. The full
    NetworkX graph is untouched and used only for analysis.
    """
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
                f"Matched on {', '.join(best['matched_items'])}; "
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


def evaluate_counterpart(
    graph: nx.Graph,
    root_did: str,
    counterpart: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    candidate_did = counterpart["did"]
    if not candidate_did or candidate_did == root_did:
        return None

    path = shortest_social_path(graph, root_did, candidate_did, max_hops=MAX_HOPS)
    if not path:
        return None

    hop = len(path) - 1
    jaccard = jaccard_similarity(graph, root_did, candidate_did)

    return {
        "did": candidate_did,
        "handle": counterpart["handle"],
        "role": counterpart["role"],
        "matched_items": counterpart["matched_items"],
        "hop": hop,
        "path": path,
        "jaccard": jaccard,
        "score": round(jaccard * 100),
        "score_badge": score_badge_for(jaccard, hop),
        "post_id": counterpart["post_id"],
        "post_text": counterpart["post_text"],
    }


def rank_counterparts(evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lowest hop is nearest; Jaccard breaks ties among equally near candidates."""
    return sorted(
        evaluations,
        key=lambda item: (
            int(item.get("hop", MAX_HOPS + 1)),
            -float(item.get("jaccard", 0.0)),
        ),
    )


@router.get("/graph/{user_did}")
def get_sna_graph(
    user_did: str, hop: int = DEFAULT_HOP_VIEW, post_id: Optional[str] = None
):
    if hop not in (1, 2, 3):
        raise HTTPException(status_code=422, detail="hop must be 1, 2, or 3.")

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

    # 1. Build the actual NetworkX social graph.
    graph = build_social_graph(root_did, max_hops=MAX_HOPS)

    # 2. Resource-compatible counterpart candidates work in either direction.
    target_items = needed_items if target_role == "Victim" else offered_items
    counterparts = (
        get_item_matched_counterparts(target_role, target_items) if target_items else []
    )
    counterpart_lookup = {item["did"]: item for item in counterparts if item.get("did")}

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

    root_doc = get_user_document(root_did)
    root_followers = as_string_set(root_doc.get("followers"))
    root_following = as_string_set(root_doc.get("following"))
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

    # 4. Build all three dashboard views from the same NetworkX graph.
    hop_views = {
        str(level): build_path_view(
            graph, root_did, level, target_role, target_post, counterpart_lookup,
            ranked_matches, root_needed_items=needed_items,
        )
        for level in (1, 2, 3)
    }
    selected_view = hop_views[str(hop)]

    return {
        "target": {
            "did": root_did,
            "handle": root_handle,
            "role": target_role,
            "urgency_level": normalize_urgency(
                target_post.get("urgency_level") or target_post.get("urgency")
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
