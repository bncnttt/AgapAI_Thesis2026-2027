import time
from datetime import datetime, timezone

from agapai_api.clients import client, db, mongo_connected
from agapai_api.config import DEFAULT_GRAPH_MEMBER_LIMIT, GRAPH_PAGE_LIMIT, SEARCH_POSTS_PAGE_LIMIT
from agapai_api.corpus import get_multilingual_disaster_terms
from agapai_api.geography import PSGC_CEBU_CITIES_MUNICIPALITIES, format_location_name, get_cebu_location_search_terms

def get_attr(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def search_bluesky_posts(search_term, max_posts, since=None, until=None):
    posts = []
    cursor = None
    while True:
        remaining = None if max_posts is None or max_posts < 0 else max_posts - len(posts)
        if remaining is not None and remaining <= 0:
            break

        try:
            params = {
                "q": search_term,
                "limit": SEARCH_POSTS_PAGE_LIMIT if remaining is None else min(SEARCH_POSTS_PAGE_LIMIT, remaining),
            }
            if since:
                params["since"] = since
            if until:
                params["until"] = until
            if cursor:
                params["cursor"] = cursor

            response = client.app.bsky.feed.search_posts(params=params)
        except Exception as search_err:
            error_message = repr(search_err)
            print(f"Search API Call failure for '{search_term}': {error_message}")
            raise RuntimeError(
                f"Bluesky search failed for term '{search_term}': {error_message}"
            ) from search_err

        page_posts = get_attr(response, 'posts', []) or []
        if not page_posts:
            break

        posts.extend(page_posts)
        cursor = get_attr(response, 'cursor')
        if not cursor:
            break

    return posts


def format_bluesky_date_bound(value, end_of_day=False):
    """Formats a date/datetime value into the ISO 8601 string Bluesky's search API expects for since/until."""
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))

    if parsed.tzinfo is None:
        if end_of_day:
            parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999000)
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def get_latest_saved_post_created_at(collection):
    """
    Returns the created_at of the most recently saved post in the given
    MongoDB collection (sorted descending), or None if the collection has
    no posts yet. created_at is stored as an ISO 8601 UTC string, so
    descending string sort matches chronological order.
    """
    latest = collection.find_one(sort=[("created_at", -1)])
    return latest.get("created_at") if latest else None


def resolve_incremental_since(collection, fallback_since):
    """
    True incremental retrieval: returns the created_at of our most
    recently saved post so a Bluesky search only asks for what's new,
    instead of re-fetching full history every run. Falls back to
    fallback_since (the caller's originally requested/full-range start)
    when the collection is empty -- e.g. the very first run.

    The returned boundary is inclusive of that latest post, so it will be
    fetched again; the existing URI-based dedup check is what's expected
    to silently skip it, rather than this function nudging the boundary
    forward and risking missing a post with an identical timestamp.
    """
    latest_created_at = get_latest_saved_post_created_at(collection)
    return latest_created_at if latest_created_at else fallback_since


def get_reply_parent_uri(post_view):
    """Extracts reply.parent.uri from a post's record, if this post is a reply."""
    record = get_attr(post_view, "record")
    reply_ref = get_attr(record, "reply") if record else None
    parent_ref = get_attr(reply_ref, "parent") if reply_ref else None
    return get_attr(parent_ref, "uri") if parent_ref else None


def resolve_parent_post_texts(parent_uris):
    """
    Batch-resolves Bluesky post URIs to their text content, for thread
    stitching: a reply like "It's completely underwater!" means nothing
    without the parent post it's replying to. Uses app.bsky.feed.getPosts,
    which accepts up to 25 URIs per call.
    """
    unique_uris = sorted({uri for uri in parent_uris if uri})
    if not unique_uris:
        return {}

    resolved = {}
    batch_size = 25
    for batch_start in range(0, len(unique_uris), batch_size):
        batch = unique_uris[batch_start:batch_start + batch_size]
        try:
            response = client.app.bsky.feed.get_posts(params={"uris": batch})
        except Exception as fetch_error:
            print(f"[AGAPAI LOG] WARNING: Failed to resolve {len(batch)} parent post(s): {fetch_error}")
            continue

        for post in get_attr(response, "posts", []) or []:
            uri = get_attr(post, "uri")
            record = get_attr(post, "record")
            text = get_attr(record, "text", "") if record else ""
            if uri:
                resolved[uri] = text

    return resolved


def search_cebu_location_posts(start_date, end_date, max_posts=-1, location_terms=None):
    """
    Retrieves Bluesky posts created strictly within [start_date, end_date]
    that mention a dynamically extracted, Cebu-anchored PSGC location term
    (e.g. "Talisay Cebu"), so foreign homonyms of the same place name are
    not fetched.
    """
    since_value = format_bluesky_date_bound(start_date)
    until_value = format_bluesky_date_bound(end_date, end_of_day=True)

    terms = location_terms if location_terms is not None else get_cebu_location_search_terms()

    print(f"[AGAPAI LOG] Querying Bluesky API for date range: {start_date} to {end_date}...")

    posts_by_uri = {}
    for term in terms:
        for post in search_bluesky_posts(term, max_posts, since=since_value, until=until_value):
            post_uri = get_attr(post, "uri")
            if post_uri:
                posts_by_uri[post_uri] = post

    return list(posts_by_uri.values())


def build_dialect_aware_cebu_queries(location_names=None, multilingual_terms=None, max_queries=None):
    """
    Combines dynamically extracted PSGC Cebu location names with the
    dynamically generated multilingual disaster-term corpus into targeted
    search queries of the form "{location} Cebu {multilingual_term}"
    (e.g. "Mandaue Cebu baha", "Talisay Cebu sunog").
    """
    location_names = location_names if location_names is not None else sorted(PSGC_CEBU_CITIES_MUNICIPALITIES)
    multilingual_terms = multilingual_terms if multilingual_terms is not None else get_multilingual_disaster_terms()

    queries = sorted({
        f"{format_location_name(location)} Cebu {term}"
        for location in location_names
        for term in multilingual_terms
    })

    if max_queries is not None and max_queries >= 0:
        queries = queries[:max_queries]

    print(f"[AGAPAI LOG] Created {len(queries)} dialect-aware search queries for Cebu.")
    return queries


def fetch_and_persist_dialect_aware_cebu_posts(start_date, end_date, max_posts=-1, queries=None):
    """
    Executes dialect-aware, Cebu-anchored search queries against Bluesky's
    app.bsky.feed.searchPosts for posts created strictly within
    [start_date, end_date], then immediately upserts every raw post payload
    (matched on URI) into both the MongoDB 'raw_posts' and 'posts'
    collections so nothing fetched is lost or duplicated.
    """
    since_value = format_bluesky_date_bound(start_date)
    until_value = format_bluesky_date_bound(end_date, end_of_day=True)

    search_queries = queries if queries is not None else build_dialect_aware_cebu_queries()

    posts_by_uri = {}
    for search_query in search_queries:
        for post in search_bluesky_posts(search_query, max_posts, since=since_value, until=until_value):
            post_uri = get_attr(post, "uri")
            if post_uri:
                posts_by_uri[post_uri] = post

    saved_count = 0
    if mongo_connected and db is not None:
        raw_posts_col = db["raw_posts"]
        posts_col = db["posts"]

        for post_uri, post in posts_by_uri.items():
            record = get_attr(post, "record")
            post_text = get_attr(record, "text", "") if record else get_attr(post, "text", "")
            author = get_attr(post, "author")

            raw_post_document = {
                "uri": post_uri,
                "cid": get_attr(post, "cid"),
                "author_did": get_attr(author, "did"),
                "author_handle": get_attr(author, "handle"),
                "text": post_text,
                "created_at": get_attr(record, "created_at") if record else None,
                "retrieval_source": "dialect_aware_cebu_search",
                "date_range_start": start_date,
                "date_range_end": end_date,
            }

            raw_posts_col.update_one({"uri": post_uri}, {"$set": raw_post_document}, upsert=True)
            posts_col.update_one({"_id": post_uri}, {"$set": raw_post_document}, upsert=True)
            saved_count += 1

    print(f"[AGAPAI LOG] Saved {saved_count} raw posts into MongoDB 'raw_posts'.")

    return list(posts_by_uri.values())


def _call_with_rate_limit_retry(fn, *args, max_retries=3, base_delay_seconds=2, **kwargs):
    """
    Generic backoff-and-retry wrapper for Bluesky API calls. The cascade
    relevance check (get_author_bio_text/get_recent_post_texts) can issue
    one extra profile/timeline call per candidate post whose own text has
    no Cebu clue, so it's more exposed to rate limiting than a plain
    search call. Retries only on a rate-limit-shaped failure; any other
    exception propagates immediately.
    """
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as call_error:
            is_rate_limited = "RateLimit" in repr(call_error) or "429" in repr(call_error)
            if not is_rate_limited or attempt == max_retries - 1:
                raise
            time.sleep(base_delay_seconds * (2 ** attempt))


def get_author_bio_text(actor):
    """
    Fetches an author's profile bio/description text -- cascade Level 2.
    A post with no location clue in its own text can still be confidently
    Cebu-relevant if the author's bio names a Cebu place.
    """
    if not actor:
        return ""
    try:
        profile = _call_with_rate_limit_retry(
            client.app.bsky.actor.get_profile, params={"actor": actor}
        )
    except Exception as profile_error:
        print(f"[AGAPAI LOG] WARNING: Failed to fetch bio for '{actor}': {profile_error}")
        return ""
    return get_attr(profile, "description", "") or ""


def get_recent_post_texts(actor, limit=5):
    """
    Fetches an author's most recent post texts -- cascade Level 3, the
    last and most expensive signal checked only when neither the post
    itself nor the author's bio names a Cebu place.
    """
    if not actor:
        return []
    try:
        feed_response = _call_with_rate_limit_retry(
            client.app.bsky.feed.get_author_feed, params={"actor": actor, "limit": limit}
        )
    except Exception as feed_error:
        print(f"[AGAPAI LOG] WARNING: Failed to fetch recent posts for '{actor}': {feed_error}")
        return []

    texts = []
    for item in get_attr(feed_response, "feed", []) or []:
        post = get_attr(item, "post")
        record = get_attr(post, "record") if post else None
        text = get_attr(record, "text", "") if record else ""
        if text:
            texts.append(text)
    return texts


def collect_graph_members(fetch_method, actor, collection_name, max_members=DEFAULT_GRAPH_MEMBER_LIMIT):
    members = []
    cursor = None
    while True:
        try:
            response = fetch_method(
                params={"actor": actor, "limit": GRAPH_PAGE_LIMIT, "cursor": cursor}
            )
            page_members = get_attr(response, collection_name, []) or []
            for member in page_members:
                member_did = get_attr(member, 'did')
                member_handle = get_attr(member, 'handle')
                if member_did:
                    members.append(member_did)
                elif member_handle:
                    members.append(member_handle)
                if max_members is not None and max_members > 0 and len(members) >= max_members:
                    return members
            cursor = get_attr(response, 'cursor')
            if not cursor or not page_members:
                break
        except Exception:
            break
    return members


def collect_graph_members_with_fallback(fetch_method, actor_did, actor_handle, collection_name, max_members):
    members = []
    seen_members = set()
    seen_actors = set()
    actors_to_try = [actor_did, actor_handle]
    for actor in actors_to_try:
        if not actor or actor in seen_actors:
            continue
        seen_actors.add(actor)
        actor_members = collect_graph_members(fetch_method, actor, collection_name, max_members)
        for member in actor_members:
            if member not in seen_members:
                members.append(member)
                seen_members.add(member)
            if max_members is not None and max_members > 0 and len(members) >= max_members:
                return members
    return members



def empty_social_graph():
    return {
        "follower_count": 0,
        "following_count": 0,
        "followers": [],
        "following": [],
        "mutual_ties": [],
    }


def compute_interaction_ties(author_did, author_handle, posts_col):
    """
    Fallback graph built from the collected dataset when follow lists are
    unavailable (rate limits): accounts this author @mentioned or replied to,
    and accounts that @mentioned or replied to them, restricted to collected users.
    """
    if posts_col is None or not author_did:
        return []

    known = {}
    for doc in posts_col.find({}, {"author_did": 1, "author_handle": 1}):
        did, handle = doc.get("author_did"), doc.get("author_handle")
        if did:
            known[did] = handle
    handle_to_did = {h.lower(): d for d, h in known.items() if h}
    own_handle = (author_handle or "").lower()

    def interactions(doc):
        found = set()
        text = doc.get("text") or ""
        for token in text.split():
            if token.startswith("@"):
                did = handle_to_did.get(token[1:].strip(".,;:!?()").lower())
                if did:
                    found.add(did)
        parent = doc.get("reply_parent_uri") or ""
        if parent.startswith("at://"):
            found.add(parent[5:].split("/", 1)[0])
        return found

    ties = set()
    for doc in posts_col.find({}, {"author_did": 1, "text": 1, "reply_parent_uri": 1}):
        poster = doc.get("author_did")
        targets = interactions(doc)
        if poster == author_did:
            ties |= targets
        elif author_did in targets or (own_handle and f"@{own_handle}" in (doc.get("text") or "").lower()):
            ties.add(poster)
    ties.discard(author_did)
    return sorted(t for t in ties if t in known)


def hydrate_social_graph(author_did, author_handle, member_limit=None, posts_col=None):
    """
    Build the social_graph dict for one author from Bluesky: official counts
    from getProfile, follower/following lists from getFollowers/getFollows,
    mutual_ties = mutual follows (plus dataset interaction ties as a fallback
    or supplement). Never raises; returns the empty graph on total failure.
    """
    graph = empty_social_graph()
    graph["display_name"] = None
    actor = author_did or author_handle
    if not actor:
        return graph

    try:
        profile = _call_with_rate_limit_retry(
            client.app.bsky.actor.get_profile, params={"actor": actor}
        )
        graph["follower_count"] = int(get_attr(profile, "followers_count", 0) or 0)
        graph["following_count"] = int(get_attr(profile, "follows_count", 0) or 0)
        graph["display_name"] = get_attr(profile, "display_name", None)
    except Exception as error:
        print(f"[AGAPAI LOG] WARNING: getProfile failed for '{actor}': {error}")

    for key, method, name in (
        ("following", client.app.bsky.graph.get_follows, "follows"),
        ("followers", client.app.bsky.graph.get_followers, "followers"),
    ):
        if member_limit == 0:
            break
        try:
            graph[key] = _call_with_rate_limit_retry(
                collect_graph_members_with_fallback,
                method, author_did, author_handle, name, member_limit,
            )
        except Exception as error:
            print(f"[AGAPAI LOG] WARNING: {name} fetch failed for '{actor}': {error}")

    mutual = set(graph["followers"]) & set(graph["following"])
    try:
        mutual |= set(compute_interaction_ties(author_did, author_handle, posts_col))
    except Exception as error:
        print(f"[AGAPAI LOG] WARNING: interaction fallback failed for '{actor}': {error}")
    graph["mutual_ties"] = sorted(mutual)

    # If the profile call failed but lists came back, use the list sizes.
    graph["follower_count"] = graph["follower_count"] or len(graph["followers"])
    graph["following_count"] = graph["following_count"] or len(graph["following"])
    return graph


def backfill_social_graphs(posts_col, users_col, member_limit=100, limit=None, pause_seconds=0.3):
    """
    Update social_graph on posts and the users collection for every author
    whose stored follower_count is 0 or missing (checked on both collections).
    Returns a summary dict.
    """
    authors = {}
    post_query = {"$or": [
        {"social_graph.follower_count": 0},
        {"social_graph.follower_count": {"$exists": False}},
    ]}
    for doc in posts_col.find(post_query, {"author_did": 1, "author_handle": 1}):
        if doc.get("author_did"):
            authors.setdefault(doc["author_did"], doc.get("author_handle"))
    if users_col is not None:
        user_query = {"$or": [{"follower_count": 0}, {"follower_count": {"$exists": False}}]}
        for doc in users_col.find(user_query, {"handle": 1}):
            authors.setdefault(doc["_id"], doc.get("handle"))
    if limit:
        authors = dict(list(authors.items())[:limit])

    updated = skipped = 0
    for did, handle in authors.items():
        graph = hydrate_social_graph(did, handle, member_limit, posts_col)
        display_name = graph.pop("display_name", None)
        if not (graph["follower_count"] or graph["following_count"] or graph["mutual_ties"]):
            skipped += 1
            continue
        posts_col.update_many({"author_did": did}, {"$set": {"social_graph": graph}})
        if users_col is not None:
            fields = {
                "follower_count": graph["follower_count"],
                "following_count": graph["following_count"],
                "mutual_tie_count": len(graph["mutual_ties"]),
                "followers": graph["followers"],
                "following": graph["following"],
                "mutual_ties": graph["mutual_ties"],
            }
            if handle:
                fields["handle"] = handle
            if display_name:
                fields["display_name"] = display_name
            users_col.update_one({"_id": did}, {"$set": fields}, upsert=True)
        updated += 1
        time.sleep(pause_seconds)
    return {"authors_found": len(authors), "updated": updated, "skipped": skipped}
