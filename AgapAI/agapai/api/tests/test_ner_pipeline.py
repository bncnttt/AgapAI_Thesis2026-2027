"""Scoped pipeline integration tests using an isolated local MongoDB database.

Run from api: ../.venv/Scripts/python -m unittest discover -s tests -v
Only model entity recognition is stubbed; location/date rules and Mongo queries run.
"""
import importlib
import os
import re
import sys
import unittest
from datetime import datetime
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from bson import ObjectId
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pymongo import MongoClient
import spacy

from agapai_api import ner_algorithm


tokenizer = spacy.blank("tl")


def recognize_entities(text):
    doc = tokenizer(text)
    doc.ents = [doc.char_span(match.start(), match.end(), label="LOC")
                for match in re.finditer(r"Poblacion, Alcantara|Poblacion|Alcantara|Talisay|Pardo|Cebu", text)]
    return doc


def post(post_id, text="Baha sa Talisay", **fields):
    return {
        "_id": post_id, "text": text, "created_at": "2026-10-28T12:00:00Z",
        "disaster_type": "flood", "author_handle": "reporter.bsky.social",
        **fields,
    }


class NerPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mongo = MongoClient(
            os.getenv("AGAPAI_TEST_MONGO_URI", "mongodb://localhost:27017/"),
            serverSelectionTimeoutMS=3000,
        )
        cls.mongo.admin.command("ping")
        cls.database_name = "agapai_ner_tests_" + uuid4().hex
        cls.posts = cls.mongo[cls.database_name].posts
        # Avoid application startup, Bluesky login, and classifier initialization.
        clients = ModuleType("agapai_api.clients")
        clients.posts_col, clients.mongo_connected = cls.posts, True
        pipeline = ModuleType("agapai_api.pipeline")
        pipeline.DISASTER_TYPES = ["flood", "earthquake", "typhoon", "fire", "landslide"]
        with patch.dict(sys.modules, {
            "agapai_api.clients": clients, "agapai_api.pipeline": pipeline,
        }):
            cls.routes = importlib.import_module("agapai_api.ner_routes")
        app = FastAPI()
        app.include_router(cls.routes.router)
        cls.api = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.api.close()
        cls.mongo.drop_database(cls.database_name)
        cls.mongo.close()

    def setUp(self):
        self.posts.delete_many({})
        model = patch.object(ner_algorithm, "get_nlp", return_value=recognize_entities)
        model.start()
        self.addCleanup(model.stop)
        route_model = patch.object(self.routes, "get_nlp", return_value=recognize_entities)
        route_model.start()
        self.addCleanup(route_model.stop)
        translator = patch.object(ner_algorithm, "_translate_temporal_text", return_value=None)
        translator.start()
        self.addCleanup(translator.stop)
        self.bio = patch.object(self.routes, "_fetch_author_bio", return_value="")
        self.real_fetch_history = self.routes._fetch_previous_post_texts
        self.history = patch.object(self.routes, "_fetch_previous_post_texts", return_value=[])
        self.fetch_bio = self.bio.start()
        self.fetch_history = self.history.start()
        self.addCleanup(self.bio.stop)
        self.addCleanup(self.history.stop)

    def process(self, ids):
        response = self.api.post("/ner/process-all?limit=100", json={"post_ids": ids})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def results(self, ids):
        response = self.api.post("/ner/results", json={"post_ids": ids})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_all_loaded_posts_processed_across_batches_and_views_agree(self):
        ids = [f"at://loaded/{index}" for index in range(101)]
        self.posts.insert_many([post(value) for value in ids] + [post("outside")])
        self.posts.update_one({"_id": ids[0]}, {"$set": {"text": "Baha sa Talisay noong 2026-10-27"}})
        self.assertEqual(self.process(ids), {"processed": 100, "remaining": 1, "batch_size": 100})
        self.assertEqual(self.process(ids)["processed"], 1)
        data = self.results(ids)
        self.assertEqual(data["total"], 101)
        self.assertEqual({row["_id"] for row in data["rows"]}, set(ids))
        self.assertNotIn("ner_processed_at", self.posts.find_one({"_id": "outside"}))
        event = data["events"][0]
        self.assertEqual(event["post_count"], 101)
        self.assertEqual({p["_id"] for p in event["posts"]}, set(ids))
        self.assertTrue(all(row["event_id"] == event["event_id"] for row in data["rows"]))
        self.assertEqual(event["event_date"], "2026-10-27")
        fallback = next(row for row in data["rows"] if row["_id"] == ids[1])
        self.assertEqual(fallback["ner_event_date"], "2026-10-28")

    def test_cached_posts_and_changed_text_without_optional_location(self):
        self.posts.insert_one(post("loaded"))
        self.process(["loaded"])
        with patch.object(self.routes, "_process_post", wraps=self.routes._process_post) as extract:
            self.assertEqual(self.process(["loaded"])["processed"], 0)
            extract.assert_not_called()
            self.posts.update_one({"_id": "loaded"}, {"$set": {"text": "Baha sa Poblacion, Alcantara"}})
            self.assertEqual(self.process(["loaded"])["processed"], 1)
        location = self.results(["loaded"])["rows"][0]["ner_locations"][0]
        self.assertEqual(location["breadcrumb"][-2], "Alcantara")

    def test_empty_selection_never_processes_or_returns_other_posts(self):
        self.posts.insert_one(post("outside"))
        self.assertEqual(self.process([])["processed"], 0)
        self.assertEqual(self.results([]), {"rows": [], "events": [], "total": 0})
        self.assertNotIn("ner_processed_at", self.posts.find_one())
        self.assertEqual(self.api.post("/ner/results").status_code, 422)
        self.assertEqual(self.api.post("/ner/results", json={}).status_code, 422)

    def test_later_batch_resolves_earlier_ambiguity_and_mixed_timestamp_types(self):
        documents = [post("ambiguous", "Baha sa Poblacion", created_at=datetime(2026, 10, 28, 12))]
        documents += [post(f"filler-{index}", "Baha") for index in range(99)]
        documents += [post("zz-anchor", "Baha sa Poblacion, Alcantara")]
        self.posts.insert_many(documents)
        ids = [doc["_id"] for doc in documents]
        self.assertEqual(self.process(ids)["remaining"], 1)
        self.assertTrue(self.posts.find_one({"_id": "ambiguous"})["ner_ambiguous_locations"])
        self.assertEqual(self.process(ids)["remaining"], 0)
        row = next(row for row in self.results(ids)["rows"] if row["_id"] == "ambiguous")
        self.assertEqual(row["ner_ambiguous_locations"], [])
        self.assertEqual(row["ner_locations"][0]["resolution_method"], "spatiotemporal_centroid")
        self.assertEqual(row["ner_locations"][0]["confidence"], "HIGH")
        self.assertEqual(row["ner_locations"][0]["breadcrumb"][-2], "Alcantara")

    def test_context_excludes_outside_posts_and_anchors_outside_time_window(self):
        self.posts.insert_many([
            post("ambiguous", "Baha sa Poblacion"),
            post("outside", "Baha sa Poblacion, Alcantara"),
            post("old", "Baha sa Poblacion, Alcantara", created_at="2026-10-27T12:00:00Z"),
        ])
        self.process(["outside"])
        self.process(["ambiguous", "old"])
        row = next(row for row in self.results(["ambiguous", "old"])["rows"] if row["_id"] == "ambiguous")
        self.assertEqual(row["ner_locations"], [])
        self.assertTrue(row["ner_ambiguous_locations"])

    def test_changed_selection_recomputes_previously_inferred_location(self):
        self.posts.insert_many([post("ambiguous", "Baha sa Poblacion"), post("anchor", "Baha sa Poblacion, Alcantara")])
        self.process(["ambiguous", "anchor"])
        self.assertTrue(self.posts.find_one({"_id": "ambiguous"})["ner_locations"])
        self.process(["ambiguous"])
        data = self.results(["ambiguous"])
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["rows"][0]["ner_locations"], [])
        self.assertTrue(data["rows"][0]["ner_ambiguous_locations"])
        self.assertEqual(data["events"], [])

    def test_imported_object_ids_and_legacy_processing(self):
        imported_id = ObjectId()
        self.posts.insert_many([post(imported_id), post("legacy")])
        self.assertEqual(self.process([str(imported_id)])["processed"], 1)
        self.assertEqual(self.results([str(imported_id)])["rows"][0]["_id"], str(imported_id))
        response = self.api.post("/ner/process-all?limit=100")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["processed"], 1)

    def test_old_extraction_version_is_reprocessed_without_a_text_change(self):
        self.posts.insert_one(post("loaded", "Baha sa Brgy. Pardo at 12 PM"))
        self.process(["loaded"])
        self.posts.update_one({"_id": "loaded"}, {"$set": {
            "ner_algorithm_version": "old-version", "ner_locations": [], "ner_event_time": "08:09:33 AM",
        }})
        self.assertEqual(self.process(["loaded"])["processed"], 1)
        data = self.results(["loaded"])
        row = data["rows"][0]
        self.assertEqual(row["ner_locations"][0]["matched_name"], "Pardo")
        self.assertEqual(row["ner_ambiguous_locations"], [])
        self.assertEqual(row["ner_event_time"], "12:00:00 PM")
        self.assertEqual(data["events"][0]["event_time"], row["ner_event_time"])
        self.assertEqual(self.process(["loaded"])["processed"], 0)

    def test_route_uses_bio_before_history_and_post_before_both(self):
        self.posts.insert_one(post("direct", "Baha sa Pardo"))
        self.process(["direct"])
        self.fetch_bio.assert_not_called()
        self.fetch_history.assert_not_called()
        self.posts.insert_one(post("bio", "Help please"))
        self.fetch_bio.return_value = "Pardo"
        self.process(["bio"])
        self.assertEqual(self.results(["bio"])["rows"][0]["ner_locations"][0]["confidence"], "MEDIUM")
        self.fetch_history.assert_not_called()
        self.fetch_bio.return_value = "Volunteer"
        self.fetch_history.return_value = ["No location", "Pardo"]
        self.posts.insert_one(post("history", "Help please"))
        self.process(["history"])
        self.assertEqual(self.results(["history"])["rows"][0]["ner_locations"][0]["confidence"], "LOW")

    def test_history_filters_current_future_other_authors_and_counts_ten(self):
        def item(uri, time, author="did:test", text="Help"):
            return {"post": {"uri": uri, "author": {"did": author}, "record": {"created_at": time, "text": text}}}
        feed = [item("current", "2026-10-28T12:00:00Z"), item("future", "2026-10-29T12:00:00Z"),
                item("repost", "2026-10-27T12:00:00Z", author="did:other")]
        feed += [item(f"past-{i}", f"2026-10-{27-i:02d}T12:00:00Z", text=f"Post {i}") for i in range(11)]
        bluesky = ModuleType("agapai_api.bluesky")
        bluesky.get_attr = lambda obj, key, default=None: obj.get(key, default) if isinstance(obj, dict) else default
        bluesky.client = SimpleNamespace(app=SimpleNamespace(bsky=SimpleNamespace(feed=SimpleNamespace(get_author_feed=object()))))
        bluesky._call_with_rate_limit_retry = lambda *args, **kwargs: {"feed": feed}
        with patch.dict(sys.modules, {"agapai_api.bluesky": bluesky}):
            texts = self.real_fetch_history(post("current", author_did="did:test"))
        self.assertEqual(texts, [f"Post {i}" for i in range(10)])

    def test_ambiguous_post_does_not_request_author_fallback(self):
        self.fetch_bio.return_value = "Alcantara"
        result = self.routes._process_post(post("ambiguous", "Baha sa Poblacion"))
        self.assertEqual(result["ner_locations"], [])
        self.assertTrue(result["ner_ambiguous_locations"])
        self.fetch_bio.assert_not_called()
        self.fetch_history.assert_not_called()

    def test_map_and_anchors_prefer_specific_place_over_province(self):
        self.posts.insert_one(post("loaded", "Cebu reports flood in Pardo"))
        self.process(["loaded"])
        data = self.results(["loaded"])
        self.assertEqual(data["events"][0]["place"], "Pardo")
        locations = data["rows"][0]["ner_locations"]
        self.assertEqual({loc["matched_name"] for loc in locations}, {"Cebu", "Pardo"})
        self.assertEqual([loc["matched_name"] for loc in self.routes._direct_anchor_locations(locations)], ["Pardo"])


if __name__ == "__main__":
    unittest.main()
