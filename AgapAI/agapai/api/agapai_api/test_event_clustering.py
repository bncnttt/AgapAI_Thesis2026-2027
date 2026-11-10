from datetime import datetime, timedelta
from ner_event_clustering import cluster_posts_into_events

base_time = datetime(2026, 8, 9, 14, 0, 0)

posts = [
    {"id": "post1", "disaster_type": "fire", "lat": 9.9500, "lon": 123.4281, "created_at": base_time},
    {"id": "post2", "disaster_type": "fire", "lat": 9.9505, "lon": 123.4285, "created_at": base_time + timedelta(hours=1)},
    {"id": "post3", "disaster_type": "flood", "lat": 9.9500, "lon": 123.4281, "created_at": base_time},
    {"id": "post4", "disaster_type": "fire", "lat": 10.3157, "lon": 123.8854, "created_at": base_time},
    {"id": "post5", "disaster_type": "fire", "lat": 9.9502, "lon": 123.4283, "created_at": base_time + timedelta(hours=10)},
]

events = cluster_posts_into_events(posts)

for i, event in enumerate(events):
    print("Event " + str(i) + ":")
    for post in event:
        print("  " + post["id"] + " - " + post["disaster_type"] + " at (" + str(post["lat"]) + ", " + str(post["lon"]) + ")")
    print()
