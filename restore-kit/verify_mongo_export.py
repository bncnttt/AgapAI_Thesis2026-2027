"""Verify a MongoDB Extended JSON export against its manifest and live database."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bson import json_util
from bson.json_util import CANONICAL_JSON_OPTIONS
from pymongo import MongoClient


EXPECTED_COUNTS = {"events": 1, "posts": 60, "users": 22}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_id(value: object) -> str:
    return json_util.dumps(value, json_options=CANONICAL_JSON_OPTIONS, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Export folder")
    parser.add_argument("--db", default="AgapAI_Database_Final", help="MongoDB database")
    parser.add_argument("--uri", default="mongodb://localhost:27017/", help="MongoDB URI")
    args = parser.parse_args()

    output_root = args.out.expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest root is not an object")
        if manifest.get("database") != args.db:
            raise ValueError("database name mismatch")
        entries = manifest.get("collections")
        if not isinstance(entries, list):
            raise ValueError("collections is not a list")
        if not isinstance(manifest.get("format"), str) or not isinstance(manifest.get("created"), str):
            raise ValueError("manifest fields are missing")
        names = [entry.get("name") for entry in entries if isinstance(entry, dict)]
        if len(names) != len(entries) or len(names) != len(set(names)):
            raise ValueError("collection entries are invalid or duplicated")
    except Exception as error:
        print(f"FAIL manifest.json ({type(error).__name__})")
        return 1

    failures = []
    try:
        with MongoClient(args.uri) as client:
            database = client[args.db]
            live_names = set(database.list_collection_names())
            manifest_names = set(names)
            if live_names != manifest_names:
                failures.append("collection_names")
            if not set(EXPECTED_COUNTS).issubset(manifest_names):
                failures.append("expected_collections")

            for entry in entries:
                name = entry["name"]
                relative = Path(entry.get("file", ""))
                file_path = (output_root / relative).resolve()
                label = relative.as_posix() if relative.parts else f"{name}.json"
                errors = []

                if relative.is_absolute() or not file_path.is_relative_to(output_root):
                    errors.append("unsafe_path")
                    file_bytes = b""
                    documents = None
                else:
                    try:
                        file_bytes = file_path.read_bytes()
                        documents = json_util.loads(file_bytes.decode("utf-8"))
                        if not isinstance(documents, list):
                            errors.append("not_array")
                            documents = None
                    except Exception as error:
                        errors.append(f"parse_{type(error).__name__}")
                        file_bytes = b""
                        documents = None

                if documents is not None:
                    if any(not isinstance(document, dict) or "_id" not in document for document in documents):
                        errors.append("missing_id")
                    if len(documents) != entry.get("count"):
                        errors.append("manifest_count")
                    if name in EXPECTED_COUNTS and len(documents) != EXPECTED_COUNTS[name]:
                        errors.append("expected_count")
                    live_collection = database[name]
                    live_count = live_collection.count_documents({})
                    if len(documents) != live_count:
                        errors.append("live_count")
                    exported_ids = {
                        canonical_id(document["_id"])
                        for document in documents
                        if isinstance(document, dict) and "_id" in document
                    }
                    live_ids = {
                        canonical_id(document["_id"])
                        for document in live_collection.find({}, {"_id": 1})
                    }
                    if exported_ids != live_ids:
                        errors.append("id_set")

                if len(file_bytes) != entry.get("size_bytes"):
                    errors.append("size_bytes")
                if sha256_bytes(file_bytes).lower() != str(entry.get("sha256", "")).lower():
                    errors.append("sha256")
                if errors:
                    failures.append(label)
                    print(f"FAIL {label} ({', '.join(errors)})")
                else:
                    print(f"PASS {label}")

            if failures:
                raise RuntimeError("verification checks failed")
    except Exception as error:
        print(f"FAIL manifest.json ({type(error).__name__})")
        return 1

    print("PASS manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
