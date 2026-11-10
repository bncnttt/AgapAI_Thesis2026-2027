import os
import time

import pymongo
from atproto import Client

from agapai_api.config import BLUESKY_HANDLE, BLUESKY_PASSWORD

client = Client()
mongo_client = None
db = None
posts_col = None
users_col = None
mongo_connected = False
bluesky_authenticated = False
bluesky_auth_error = None


BLUESKY_LOGIN_ATTEMPTS = int(os.getenv("BLUESKY_LOGIN_ATTEMPTS", "3"))
BLUESKY_LOGIN_RETRY_DELAY = float(os.getenv("BLUESKY_LOGIN_RETRY_DELAY", "2"))


def authenticate_bluesky():
    """Log in to Bluesky, retrying transient network failures (DNS/TLS/timeouts)."""
    global bluesky_authenticated, bluesky_auth_error

    if not BLUESKY_HANDLE or not BLUESKY_PASSWORD:
        bluesky_authenticated = False
        bluesky_auth_error = "BLUESKY_HANDLE and BLUESKY_PASSWORD must be set."
        print(f"CRITICAL WARNING: Could not authenticate with Bluesky: {bluesky_auth_error}")
        return False

    for attempt in range(1, BLUESKY_LOGIN_ATTEMPTS + 1):
        try:
            print(f"Attempting API login for {BLUESKY_HANDLE} (try {attempt}/{BLUESKY_LOGIN_ATTEMPTS})...")
            client.login(BLUESKY_HANDLE, BLUESKY_PASSWORD)
            bluesky_authenticated = True
            bluesky_auth_error = None
            print("Login successful! Network connectivity verified.")
            return True
        except Exception as e:
            bluesky_authenticated = False
            cause = e.__cause__ or e.__context__
            bluesky_auth_error = repr(e) + (f" (caused by {cause!r})" if cause else "")
            # Bad credentials will not fix themselves; only retry network errors.
            if type(e).__name__ != "NetworkError" or attempt == BLUESKY_LOGIN_ATTEMPTS:
                break
            time.sleep(BLUESKY_LOGIN_RETRY_DELAY * attempt)

    print(f"CRITICAL WARNING: Could not authenticate with Bluesky: {bluesky_auth_error}")
    print("The app keeps running; Bluesky calls re-try login on demand. "
          "Check your internet/VPN/firewall and bsky.social reachability.")
    return False


def is_bluesky_authenticated():
    return bluesky_authenticated


def get_bluesky_auth_error():
    return bluesky_auth_error


def ensure_post_indexes():
    if not mongo_connected or posts_col is None:
        return

    index_fields = [
        "disaster_type",
        "created_at",
        "ner_processed_at",
        "ner_locations.latitude",
        "ner_locations.longitude",
    ]

    for field in index_fields:
        posts_col.create_index(field)

try:
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    mongo_client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
    db = mongo_client["AgapAI_Database_Final"]
    mongo_client.admin.command("ping")

    existing_collections = db.list_collection_names()
    if "posts" not in existing_collections:
        db.create_collection("posts")
    if "users" not in existing_collections:
        db.create_collection("users")

    posts_col = db["posts"]
    users_col = db["users"]
    mongo_connected = True
    print("STATUS: Connected to MongoDB database AgapAI_Database_Final with posts and users collections.")
except Exception as e:
    print(f"CRITICAL WARNING: MongoDB Connection Failed: {e}")
