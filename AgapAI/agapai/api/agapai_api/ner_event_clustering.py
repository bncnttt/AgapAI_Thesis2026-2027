import numpy as np
from sklearn.cluster import DBSCAN
from math import radians, sin, cos, sqrt, atan2

MAX_DISTANCE_KM = 2.0  # typical barangay-level radius for a single incident in urban Cebu
MAX_TIME_HOURS = 6.0  # typical window for related eyewitness reports of one event


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def is_same_event(post_a, post_b):
    if post_a["disaster_type"] != post_b["disaster_type"]:
        return False

    distance_km = haversine_km(
        post_a["lat"], post_a["lon"],
        post_b["lat"], post_b["lon"],
    )
    if distance_km > MAX_DISTANCE_KM:
        return False

    time_diff_hours = abs((post_a["created_at"] - post_b["created_at"]).total_seconds()) / 3600
    if time_diff_hours > MAX_TIME_HOURS:
        return False

    return True


def build_distance_matrix(posts):
    n = len(posts)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Binary adjacency: DBSCAN joins compatible pairs at distance zero.
            matrix[i, j] = matrix[j, i] = 0.0 if is_same_event(posts[i], posts[j]) else 1.0

    return matrix


def cluster_posts_into_events(posts):
    if not posts:
        return []

    distance_matrix = build_distance_matrix(posts)

    clustering = DBSCAN(eps=0.5, min_samples=1, metric="precomputed")
    labels = clustering.fit_predict(distance_matrix)

    events = {}
    for post, cluster_label in zip(posts, labels):
        events.setdefault(cluster_label, []).append(post)

    return list(events.values())

def resolve_event_datetime(event_posts):
    # "vague_date" posts (e.g. "kagabi"/"last night") carry a real date, just
    # not a time-of-day, so they anchor the event the same way explicit posts do.
    posts_with_explicit_time = [
        p for p in event_posts if p.get("datetime_source") in ("explicit", "vague_date") and p.get("event_datetime") is not None
    ]

    if posts_with_explicit_time:
        event_datetime = min(p["event_datetime"] for p in posts_with_explicit_time)

        for post in event_posts:
            if post.get("datetime_source") == "unresolved":
                post["event_datetime"] = event_datetime
                post["datetime_source"] = "inferred"

        return event_datetime

    event_datetime = min(p["created_at"] for p in event_posts)

    for post in event_posts:
        if post.get("datetime_source") == "unresolved":
            post["event_datetime"] = post["created_at"]
            post["datetime_source"] = "fallback"

    return event_datetime


def pick_event_location(event_posts):
    confidence_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

    best_post = max(event_posts, key=lambda p: confidence_rank.get(p.get("location_confidence"), 0))

    return {
        "place": best_post.get("location_text"),
        "lat": best_post["lat"],
        "lon": best_post["lon"],
    }


def pick_event_locations(event_posts):
    """Return every distinct resolved place represented in an event cluster."""
    locations = {}
    for post in event_posts:
        key = post.get("location_key") or f'{post["lat"]:.6f}:{post["lon"]:.6f}'
        location = locations.setdefault(key, {
            "id": str(key),
            "place": post.get("location_text"),
            "lat": post["lat"],
            "lon": post["lon"],
            "post_ids": [],
        })
        post_id = str(post.get("_id")) if post.get("_id") is not None else None
        if post_id is not None and post_id not in location["post_ids"]:
            location["post_ids"].append(post_id)

    for location in locations.values():
        location["post_count"] = len(location["post_ids"])
    return list(locations.values())


def build_event_summary(event_posts):
    event_datetime = resolve_event_datetime(event_posts)
    location = pick_event_location(event_posts)

    return {
        "disaster_type": event_posts[0]["disaster_type"],
        "location": location,
        "locations": pick_event_locations(event_posts),
        "event_datetime": event_datetime,
        "posts": event_posts,
        "related_users": list({p["author_handle"] for p in event_posts if "author_handle" in p}),
    }
