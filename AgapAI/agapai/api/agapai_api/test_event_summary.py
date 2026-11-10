from datetime import datetime, timedelta
from ner_event_clustering import build_event_summary

base_time = datetime(2026, 8, 9, 14, 0, 0)

event_posts = [
    {
        "author_handle": "user1.bsky.social",
        "disaster_type": "fire",
        "lat": 9.9500, "lon": 123.4281,
        "location_text": "Moalboal", "location_confidence": "HIGH",
        "created_at": base_time,
        "event_datetime": base_time,
        "datetime_source": "explicit",
    },
    {
        "author_handle": "user2.bsky.social",
        "disaster_type": "fire",
        "lat": 9.9505, "lon": 123.4285,
        "location_text": "Moalboal", "location_confidence": "MEDIUM",
        "created_at": base_time + timedelta(minutes=30),
        "event_datetime": None,
        "datetime_source": "unresolved",
    },
]

summary = build_event_summary(event_posts)
print("Disaster type:", summary["disaster_type"])
print("Location:", summary["location"])
print("Event datetime:", summary["event_datetime"])
print("Related users:", summary["related_users"])
print()
print("Post 2 after resolution:")
print("  event_datetime:", event_posts[1]["event_datetime"])
print("  datetime_source:", event_posts[1]["datetime_source"])
