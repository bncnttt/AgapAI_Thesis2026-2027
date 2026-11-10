"""Export one MongoDB database to Extended JSON arrays and a checksum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from bson import json_util
from pymongo import MongoClient


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Export folder")
    parser.add_argument("--db", default="AgapAI_Database_Final", help="MongoDB database")
    parser.add_argument("--uri", default="mongodb://localhost:27017/", help="MongoDB URI")
    args = parser.parse_args()

    try:
        output_root = args.out.expanduser().resolve()
        database_folder = output_root / args.db
        database_folder.mkdir(parents=True, exist_ok=True)
        with MongoClient(args.uri) as client:
            database = client[args.db]
            collection_names = sorted(database.list_collection_names())
            collection_manifest = []

            for name in collection_names:
                documents = list(database[name].find())
                relative_file = Path(args.db) / f"{name}.json"
                output_file = output_root / relative_file
                payload = json_util.dumps(documents, indent=2)
                with output_file.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(payload)

                collection_manifest.append(
                    {
                        "name": name,
                        "count": len(documents),
                        "file": relative_file.as_posix(),
                        "size_bytes": output_file.stat().st_size,
                        "sha256": sha256_file(output_file),
                    }
                )

            manifest = {
                "database": args.db,
                "format": "bson.json_util Extended JSON array",
                "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "collections": collection_manifest,
            }
            with (output_root / "manifest.json").open(
                "w", encoding="utf-8", newline="\n"
            ) as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")

        for entry in collection_manifest:
            print(f"{entry['name']}: {entry['count']}")
        return 0
    except Exception as error:
        print(f"Export failed: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
