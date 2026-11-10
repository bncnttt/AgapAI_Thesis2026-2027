import json

from agapai_api.bluesky import get_attr
from agapai_api.clients import authenticate_bluesky, client
from agapai_api.ner_routes import _process_post

uri = input("Paste the real post's AT URI: ").strip()

if not authenticate_bluesky():
    raise SystemExit("Bluesky login failed; check api/.env.")

response = client.app.bsky.feed.get_posts(params={"uris": [uri]})
posts = get_attr(response, "posts", []) or []
if not posts:
    raise SystemExit("Bluesky returned no post for that AT URI.")

view = posts[0]
record = get_attr(view, "record")
author = get_attr(view, "author")

target = {
    "_id": get_attr(view, "uri"),
    "text": get_attr(record, "text", ""),
    "author_did": get_attr(author, "did"),
    "author_handle": get_attr(author, "handle"),
    "created_at": get_attr(record, "created_at"),
}

print("Fetching target post...")
print(json.dumps(target, ensure_ascii=False, indent=2, default=str))

cache = {}
result = _process_post(target, context_cache=cache)
actor = target["author_did"] or target["author_handle"]
history_key = ("history", actor, str(target["_id"]))

print("Checking bio and history retrieval...")
print(json.dumps({
    "bio": cache.get(("bio", actor), "(not fetched; post text resolved directly)"),
    "history_count": len(cache.get(history_key, [])),
    "previous_posts": cache.get(history_key, []),
}, ensure_ascii=False, indent=2, default=str))

print("NER result...")
print(json.dumps({
    "locations": result["ner_locations"],
    "ambiguous_locations": result["ner_ambiguous_locations"],
}, ensure_ascii=False, indent=2, default=str))