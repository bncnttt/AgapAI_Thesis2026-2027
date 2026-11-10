"""Import a verified Extended JSON MongoDB export without overwriting data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from bson import json_util
from bson.json_util import CANONICAL_JSON_OPTIONS
from pymongo import MongoClient


EXPECTED_COUNTS = {"events": 1, "posts": 60, "users": 22}


def canonical_id(value: object) -> str:
    return json_util.dumps(value, json_options=CANONICAL_JSON_OPTIONS, sort_keys=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path, help="Folder containing manifest.json")
    parser.add_argument("--db", help="Target database; defaults to the manifest database")
    parser.add_argument(
        "--uri",
        default=os.getenv("MONGO_URI", "mongodb://localhost:27017/"),
        help="MongoDB URI (defaults to MONGO_URI or localhost)",
    )
    args = parser.parse_args()

    root = args.export.expanduser().resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("collections"), list):
            raise ValueError("invalid manifest")
        database_name = args.db or manifest.get("database")
        if not isinstance(database_name, str) or database_name != manifest.get("database"):
            raise ValueError("target database does not match manifest")

        entries = manifest["collections"]
        names = [entry.get("name") for entry in entries if isinstance(entry, dict)]
        if len(names) != len(entries) or len(names) != len(set(names)):
            raise ValueError("invalid or duplicate collection entries")

        prepared = []
        for entry in entries:
            name = entry.get("name")
            relative = Path(entry.get("file", ""))
            file_path = (root / relative).resolve()
            if relative.is_absolute() or not file_path.is_relative_to(root):
                raise ValueError("unsafe collection file path")
            payload = file_path.read_bytes()
            documents = json_util.loads(payload.decode("utf-8"))
            if not isinstance(documents, list):
                raise ValueError("collection file is not a JSON array")
            if len(documents) != entry.get("count"):
                raise ValueError("collection count differs from manifest")
            if name in EXPECTED_COUNTS and len(documents) != EXPECTED_COUNTS[name]:
                raise ValueError("expected collection count differs")
            if len(payload) != entry.get("size_bytes"):
                raise ValueError("collection size differs from manifest")
            if sha256_bytes(payload).lower() != str(entry.get("sha256", "")).lower():
                raise ValueError("collection SHA256 differs from manifest")
            if any(not isinstance(document, dict) or "_id" not in document for document in documents):
                raise ValueError("a document is missing _id")
            identifiers = [canonical_id(document["_id"]) for document in documents]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError("duplicate _id in collection file")
            prepared.append((name, documents, entry["count"]))

        if not set(EXPECTED_COUNTS).issubset(set(names)):
            raise ValueError("manifest is missing an expected collection")
    except Exception as error:
        print(f"FAIL export validation ({type(error).__name__})", file=sys.stderr)
        return 1

    try:
        with MongoClient(args.uri) as client:
            database = client[database_name]
            occupied = [name for name, _, _ in prepared if database[name].count_documents({}) > 0]
            if occupied:
                print("FAIL import: target collections already contain documents", file=sys.stderr)
                return 1

            for name, documents, _ in prepared:
                if documents:
                    database[name].insert_many(documents)

            passed = True
            for name, _, expected_count in prepared:
                actual_count = database[name].count_documents({})
                print(f"{name}: {actual_count}")
                if actual_count != expected_count:
                    passed = False
            if not passed:
                print("FAIL post-import count check", file=sys.stderr)
                return 1
        return 0
    except Exception as error:
        print(f"FAIL MongoDB import ({type(error).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
