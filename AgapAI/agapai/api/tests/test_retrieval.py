"""Retrieval-only tests. No credentials, live requests, model loads, or DB writes.

Run from api: ../.venv/Scripts/python -m unittest discover -s tests -p test_retrieval.py -v
"""
import ast
import copy
import io
import json
import logging
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
import pymongo

from agapai_api import corpus

# Consolidation imports the existing provider helpers. Keep their startup DB
# dependency isolated; no real credentials, database, or network are used here.
clients = ModuleType("agapai_api.clients")
clients.client, clients.db, clients.mongo_connected = Mock(), None, False
previous_clients = sys.modules.get("agapai_api.clients")
sys.modules["agapai_api.clients"] = clients
try:
    from agapai_api import retrieval as rules
finally:
    if previous_clients is None:
        sys.modules.pop("agapai_api.clients", None)
    else:
        sys.modules["agapai_api.clients"] = previous_clients

API = Path(__file__).resolve().parents[1] / "agapai_api"


def load_functions(filename, names, namespace):
    """Exercise actual orchestration without application-startup model/login side effects."""
    tree = ast.parse((API / filename).read_text(encoding="utf-8-sig"))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in selected:
        node.decorator_list = []
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, "exec"), namespace)
    return namespace


def attr(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def candidate(uri, text, author="did:test"):
    return {"uri": uri, "record": {"text": text, "created_at": "2026-10-01T12:00:00Z"},
            "author": {"did": author, "handle": "test.bsky.social"}}


def accepted_log_entries(path):
    return [json.loads(line.split("DECISION=ACCEPT ", 1)[1])
            for line in path.read_text(encoding="utf-8").splitlines() if "DECISION=ACCEPT " in line]


class Database(dict):
    name = "test"


class Collection:
    name = "posts"
    database = type("Database", (), {"name": "test"})()
    def __init__(self):
        self.docs = {}

    def distinct(self, key):
        return [doc[key] for doc in self.docs.values() if key in doc]

    def update_one(self, query, update, upsert=False):
        key = next(iter(query.values()))
        self.docs.setdefault(key, dict(query)).update(copy.deepcopy(update["$set"]))

    def insert_one(self, doc):
        self.docs[doc.get("_id", doc.get("uri"))] = copy.deepcopy(doc)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        semantic = patch.object(rules, "analyze_disaster_relevance", return_value={
            "relevant": True, "confidence": 0.95, "label": "literal disaster",
        })
        self.semantic = semantic.start()
        self.addCleanup(semantic.stop)
        temporary_logs = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_logs.cleanup)
        log_env = patch.dict(rules.os.environ, {"AGAPAI_RETRIEVAL_LOG_DIR": temporary_logs.name})
        log_env.start()
        self.addCleanup(log_env.stop)
        previous_level = rules.logger.level
        rules.logger.setLevel(logging.CRITICAL)
        self.addCleanup(rules.logger.setLevel, previous_level)
        console = next(h for h in rules.logger.handlers
                       if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler))
        quiet_console = patch.object(console, "level", logging.CRITICAL)
        quiet_console.start()
        self.addCleanup(quiet_console.stop)
        # Real explicit vocabulary, representative results of existing dynamic expansion.
        self.groups = {kind: set(words) | {kind} | set(corpus.FILIPINO_RETRIEVAL_KEYWORDS[kind])
                       for kind, words in corpus.CEBUANO_RETRIEVAL_KEYWORDS.items()}
        self.groups["fire"].add("blaze")
        self.patcher = patch.object(rules, "get_retrieval_keyword_groups", return_value=self.groups)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.bio, self.recent = Mock(return_value=""), Mock(return_value=[])
        self.run = rules.RetrievalRun(self.bio, self.recent)
        self.addCleanup(self.run.finish)

    def decide(self, text, **kwargs):
        return self.run.evaluate(text, post_id="at://test/post/1", actor="did:test", **kwargs)

    def test_semantic_rejection_overrides_keyword_and_cebu_match(self):
        self.semantic.return_value = {"relevant": False, "confidence": 0.12}
        decision = self.decide("Nag baha ug kwarta diri sa Mandaue Cebu")
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "MODEL_NOT_DISASTER")

    def test_semantic_failure_cannot_accept(self):
        self.semantic.side_effect = RuntimeError("model unavailable")
        decision = self.decide("Baha sa Mandaue Cebu, need rescue")
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "MODEL_CHECK_FAILED")

    def test_no_model_call_without_retrieval_conditions(self):
        self.decide("Hello Cebu")
        self.decide("Baha somewhere")
        self.semantic.assert_not_called()

    def test_dashboard_does_not_save_semantically_rejected_post(self):
        ns = self.route_namespace()
        self.semantic.return_value = {"relevant": False, "confidence": 0.12}
        ns["search_bluesky_posts"] = Mock(return_value=[
            candidate("metaphor", "Nag baha ug kwarta diri sa Mandaue Cebu")])
        result = ns["get_disaster_posts"](start="2026-09-01", end="2026-11-11",
                                          force_refresh=True, include_graph=False)
        self.assertEqual(result["posts_collection"], [])
        self.assertEqual(ns["posts_col"].docs, {})

    def test_all_five_disasters_with_explicit_cebu(self):
        for kind in self.groups:
            with self.subTest(kind=kind):
                d = self.decide(f"A {kind} emergency in Cebu City.")
                self.assertTrue(d["accepted"])
                self.assertIn(kind, d["disasters"])
                self.assertEqual(d["verification"], "EXPLICIT_CEBU")
        self.bio.assert_not_called()
        self.recent.assert_not_called()

    def test_every_approved_cebuano_keyword(self):
        for kind, words in corpus.CEBUANO_RETRIEVAL_KEYWORDS.items():
            for word in words:
                with self.subTest(word=word):
                    d = self.decide(f"Karon {word.upper()} sa Cebu, people need help.")
                    self.assertTrue(d["accepted"])
                    self.assertIn(kind, d["disasters"])

    def test_multilingual_and_dynamic_terms(self):
        for text in ("May lindol sa Cebu City", "Binaha ang Cebu", "A blaze in Cebu",
                     "Pagguho ng lupa sa Cebu", "Naay sunog sa Cebu, please send help"):
            with self.subTest(text=text):
                self.assertTrue(self.decide(text)["accepted"])

    def test_phrase_whitespace_and_boundaries(self):
        self.assertTrue(self.decide("MIDAHILI\n ang\t yuta sa Cebu")["accepted"])
        for text in ("bahamas Cebu", "firefly Cebu", "linear Cebu", "bagyongx Cebu"):
            self.assertFalse(self.decide(text)["accepted"])

    def test_barangay_author_bio(self):
        self.bio.return_value = "Based in Cebu City, Philippines"
        d = self.decide("Grabe ang baha sa Barangay Lahug karon.")
        self.assertTrue(d["accepted"])
        self.assertEqual(d["verification"], "AUTHOR_BIO")
        self.recent.assert_not_called()

    def test_municipality_without_cebu_uses_cascade(self):
        self.bio.return_value = "Cebu"
        self.assertEqual(self.decide("Landslide in Dalaguete. Residents need rescue.")["verification"],
                         "AUTHOR_BIO")

    def test_recent_five_one_credible_post_suffices(self):
        self.recent.return_value = ["hello", "Based in Cebu City today", "other", "other", "other"]
        d = self.decide("Nasunog ang balay sa Barangay Lahug")
        self.assertEqual(d["verification"], "AUTHOR_RECENT_POSTS")
        self.recent.assert_called_once_with("did:test", limit=5)

    def test_does_not_use_sixth_post_or_generic_cebu_mention(self):
        self.recent.return_value = ["I love Cebu City"] * 5 + ["I live in Cebu"]
        self.assertEqual(self.decide("Sunog sa Barangay Lahug")["reason"], "CEBU_NOT_VERIFIED")

    def test_generic_bio_cebu_mention_is_not_location_proof(self):
        self.bio.return_value = "I love Cebu City"
        self.assertFalse(self.decide("Sunog sa Barangay Lahug")["accepted"])

    def test_missing_disaster_skips_context(self):
        self.assertEqual(self.decide("Lunch in Cebu City")["reason"], "NO_DISASTER_MATCH")
        self.bio.assert_not_called()
        self.recent.assert_not_called()

    def test_missing_geography_skips_context_even_with_cebu_author(self):
        self.bio.return_value = "Cebu City"
        self.assertEqual(self.decide("There is a fire, please help")["reason"], "NO_GAZETTEER_LOCATION")
        self.bio.assert_not_called()
        self.recent.assert_not_called()

    def test_ambiguous_unverified(self):
        self.assertEqual(self.decide("Sunog sa San Jose")["reason"], "CEBU_NOT_VERIFIED")

    def test_shared_bio_place_is_not_cebu_proof(self):
        self.bio.return_value = "San Francisco"
        self.assertFalse(self.decide("Fire in San Jose")["accepted"])

    def test_outside_location_overrides_author(self):
        self.bio.return_value = "Cebu City"
        for text in ("Fire in San Jose, California", "Earthquake in Japan, reports from Cebu"):
            with self.subTest(text=text):
                self.assertEqual(self.decide(text)["reason"], "CONFLICTING_LOCATION")
        self.bio.assert_not_called()

    def test_existing_gazetteer_blacklist_is_respected(self):
        with patch.object(rules, "find_out_of_bounds_matches", return_value=["outside"]):
            self.assertEqual(self.decide("Fire in Cebu City")["reason"], "CONFLICTING_LOCATION")

    def test_screenshot_false_positives_and_real_house_fire(self):
        for text in ("New video out via Fire Records, San Francisco via NYC",
                     "A gyrocopter crashed in San Jose according to the fire department",
                     "New music via Fire Records in Cebu"):
            self.assertFalse(self.decide(text)["accepted"])
        self.assertTrue(self.decide("Fire department responded to a house fire in Cebu City")["accepted"])

    def test_context_failure_is_not_acceptance_and_run_continues(self):
        self.bio.side_effect = RuntimeError("private provider error")
        self.recent.side_effect = RuntimeError("private provider error")
        self.assertFalse(self.decide("Sunog sa Barangay Lahug")["accepted"])
        self.assertTrue(self.decide("Sunog sa Cebu")["accepted"])
        self.assertEqual(self.run.counts["errors"], 2)

    def test_context_cache_and_available_profile_reuse(self):
        self.recent.return_value = ["Live in Cebu"]
        for _ in range(3):
            self.assertTrue(self.decide("Baha sa Barangay Lahug", available_bio="reporter")["accepted"])
        self.bio.assert_not_called()
        self.recent.assert_called_once()

    def test_trace_and_actual_summary_counts(self):
        output = io.StringIO()
        capture = logging.StreamHandler(output)
        rules.logger.addHandler(capture)
        rules.logger.setLevel(logging.DEBUG)
        try:
            self.run.query("sunog Cebu", 2)
            self.decide("Sunog sa Cebu City")
            self.decide("hello Cebu")
            self.run.finish()
        finally:
            rules.logger.removeHandler(capture)
        trace = output.getvalue()
        for marker in ("[RETRIEVAL_TRACE]", "SEARCH_QUERY", "DISASTER_CHECK", "LOCATION_CHECK",
                       "DECISION=ACCEPT", "DECISION=REJECT", '"accepted": 1', '"rejected": 1'):
            self.assertIn(marker, trace)

    def test_expansion_preserved_and_keywords_added_to_search(self):
        with patch.object(corpus, "expand_disaster_concepts_via_wordnet", return_value=["fire", "blaze"]), \
             patch.object(corpus, "request_dialect_expansion_locally", return_value=["existing translation"]):
            terms = corpus.get_multilingual_disaster_terms()
            dialect = corpus.get_dialect_only_disaster_terms()
        self.assertTrue({"fire", "blaze", "existing translation"} <= set(terms))
        for words in corpus.CEBUANO_RETRIEVAL_KEYWORDS.values():
            self.assertTrue(set(words) <= set(terms))
            self.assertTrue(set(words) <= set(dialect))

    def route_namespace(self):
        posts, users = Collection(), Collection()
        ns = {"Optional": object, "HTTPException": HTTPException, "datetime": datetime,
              "timedelta": timedelta, "timezone": timezone, "pymongo": pymongo,
              "mongo_connected": True, "posts_col": posts, "users_col": users,
              "is_bluesky_authenticated": lambda: True, "get_attr": attr,
              "new_retrieval_run": lambda: self.run,
              "retrieval_metadata": rules.retrieval_metadata,
              "RETRIEVAL_RULE_VERSION": rules.RETRIEVAL_RULE_VERSION,
              "get_dialect_only_disaster_terms": lambda: ["sunog"],
              "get_cebu_location_search_terms": lambda: ["Cebu City"],
              "location_disaster_queries": lambda: ["landslide Dalaguete"],
              "resolve_incremental_since": lambda col, fallback_since: fallback_since,
              "extract_location_name": lambda text: "Cebu City", "DISASTER_TYPES": list(self.groups)}
        from typing import Optional
        ns["Optional"] = Optional
        load_functions("routes.py", {"get_disaster_posts", "parse_date_filter", "parse_bluesky_datetime",
                                    "is_saved_disaster_post", "process_and_store_post"}, ns)
        return ns

    def test_dashboard_acceptance_independent_of_model_and_dedup(self):
        rules.logger.setLevel(logging.DEBUG)
        ns = self.route_namespace()
        found = [candidate("valid", "Sunog sa Cebu City"), candidate("invalid", "Music in Cebu")]
        ns["search_bluesky_posts"] = Mock(return_value=found + found)
        ns["evaluate_post"] = Mock(return_value={"is_disaster": False, "disaster_type": "none",
                                                 "disaster_type_confidence": 0.1, "extracted_locations": []})
        ns["classify_role_urgency"] = Mock(return_value={"classifier_prediction": {"disaster": "none"},
            "role": None, "role_confidence": 0, "urgency": None, "urgency_confidence": 0,
            "needs": [], "donations": []})
        result = ns["get_disaster_posts"](start="2026-09-01", end="2026-11-11",
                                          force_refresh=True, include_graph=False)
        self.assertEqual(len(result["posts_collection"]), 1)
        self.assertEqual(set(ns["posts_col"].docs), {"valid"})
        ns["evaluate_post"].assert_called_once_with("Sunog sa Cebu City")
        ns["classify_role_urgency"].assert_called_once()
        self.assertEqual(self.run.counts["candidates"], 2)
        doc = ns["posts_col"].docs["valid"]
        self.assertEqual(doc["classifier_prediction"]["disaster"], "none")
        self.assertTrue(ns["is_saved_disaster_post"](doc))
        entry, = accepted_log_entries(self.run.log_path)
        self.assertEqual(entry["post_id"], "valid")
        self.assertEqual(entry["author_did"], "did:test")
        self.assertEqual(entry["author_handle"], "test.bsky.social")
        self.assertEqual(entry["text"], "Sunog sa Cebu City")
        self.assertEqual(entry["search_query"], "Cebu City")

    def test_dashboard_downstream_failure_does_not_remove_retrieval(self):
        ns = self.route_namespace()
        ns["search_bluesky_posts"] = Mock(return_value=[candidate("valid", "Sunog sa Cebu City")])
        ns["evaluate_post"] = Mock(side_effect=RuntimeError("model unavailable"))
        ns["classify_role_urgency"] = Mock(side_effect=RuntimeError("model unavailable"))
        result = ns["get_disaster_posts"](start="2026-09-01", end="2026-11-11",
                                          force_refresh=True, include_graph=False)
        self.assertEqual(len(result["posts_collection"]), 1)
        self.assertEqual(set(ns["posts_col"].docs), {"valid"})

    def test_raw_ingestion_and_dialect_helper_persist_only_valid(self):
        rules.logger.setLevel(logging.DEBUG)
        for name in ("retrieve_and_filter_cebu_disaster_posts", "retrieve_and_cascade_filter_disaster_posts",
                     "fetch_and_persist_dialect_aware_cebu_posts"):
            with self.subTest(path=name):
                raw, posts = Collection(), Collection()
                data = [candidate("valid", "Sunog sa Cebu City"), candidate("invalid", "Music in Cebu")]
                ns = {"mongo_connected": True, "db": Database(raw_posts=raw, posts=posts),
                      "datetime": datetime, "timezone": timezone,
                      "get_attr": attr, "get_author_bio_text": self.bio, "get_recent_post_texts": self.recent,
                      "get_reply_parent_uri": lambda p: None,
                      "new_retrieval_run": lambda: rules.RetrievalRun(self.bio, self.recent),
                      "retrieval_metadata": rules.retrieval_metadata,
                      "format_bluesky_date_bound": lambda v, **kw: v,
                      "resolve_incremental_since": lambda col, fallback_since: fallback_since,
                      "get_dialect_only_disaster_terms": lambda: ["sunog"],
                      "location_disaster_queries": lambda: ["landslide Dalaguete"],
                      "search_bluesky_posts": Mock(return_value=data + data)}
                file = "bluesky.py" if name.startswith("fetch") else "retrieval.py"
                load_functions(file, {name}, ns)
                kwargs = {"queries": ["sunog", "sunog Cebu"]} if name.startswith("fetch") else (
                    {"location_terms": ["Cebu"]} if name == "retrieve_and_filter_cebu_disaster_posts" else {})
                logs_before = set(self.run.log_path.parent.glob("*.log"))
                ns[name]("2026-09-01", "2026-11-11", **kwargs)
                self.assertEqual(set(raw.docs), {"valid"})
                self.assertTrue(set(posts.docs) <= {"valid"})
                log_path, = set(self.run.log_path.parent.glob("*.log")) - logs_before
                entry, = accepted_log_entries(log_path)
                self.assertEqual(entry["post_id"], "valid")
                self.assertEqual(entry["author_did"], "did:test")
                self.assertEqual(entry["author_handle"], "test.bsky.social")
                self.assertEqual(entry["text"], "Sunog sa Cebu City")
                expected_query = ("sunog" if name.startswith("fetch") else
                                  "Cebu" if "location_terms" in kwargs else "landslide Dalaguete")
                self.assertEqual(entry["search_query"], expected_query)

    def test_search_queries_without_cebu_are_bounded_and_use_registry(self):
        queries = rules.location_disaster_queries()
        self.assertIn("landslide dalaguete", queries)
        self.assertEqual(len(queries), len(corpus.DISASTER_CONCEPT_SEEDS)
                         * len(rules.PSGC_CEBU_CITIES_MUNICIPALITIES))
        self.assertLess(len(queries), 300)

    def test_url_is_not_geographic_evidence(self):
        self.assertEqual(self.decide("A fire reported https://example.com/cebu")["reason"],
                         "NO_GAZETTEER_LOCATION")

    def test_saved_model_result_cannot_override_eligibility(self):
        ns = self.route_namespace()
        self.assertTrue(ns["is_saved_disaster_post"]({"text": "Sunog sa Cebu City",
                        "is_disaster_related": False, "disaster_type": "none"}))
        self.assertFalse(ns["is_saved_disaster_post"]({"text": "Music in Cebu City",
                         "is_disaster_related": True, "disaster_type": "fire"}))

    def test_alternate_ingestion_helper_keeps_valid_on_model_failure(self):
        ns = self.route_namespace()
        ns["evaluate_post"] = Mock(side_effect=RuntimeError("unavailable"))
        ns["classify_role_urgency"] = Mock(side_effect=RuntimeError("unavailable"))
        ns["process_and_store_post"]({"_id": "valid", "text": "Sunog sa Cebu City"})
        ns["process_and_store_post"]({"_id": "invalid", "text": "Music in Cebu City"})
        self.assertEqual(set(ns["posts_col"].docs), {"valid"})
        ns["evaluate_post"].assert_called_once()

    def test_pagination_date_bounds_and_max_posts_are_preserved(self):
        search = Mock(side_effect=[{"posts": [candidate("a", "text")], "cursor": "next"},
                                   {"posts": [candidate("b", "text")]}])
        client = Mock()
        client.app.bsky.feed.search_posts = search
        ns = {"client": client, "get_attr": attr, "SEARCH_POSTS_PAGE_LIMIT": 100}
        load_functions("bluesky.py", {"search_bluesky_posts"}, ns)
        found = ns["search_bluesky_posts"]("sunog Cebu", 2, since="start", until="end")
        self.assertEqual(len(found), 2)
        self.assertEqual(search.call_args_list[0].kwargs["params"],
                         {"q": "sunog Cebu", "limit": 2, "since": "start", "until": "end"})
        self.assertEqual(search.call_args_list[1].kwargs["params"]["cursor"], "next")
        self.assertEqual(search.call_args_list[1].kwargs["params"]["limit"], 1)

    def test_recent_context_excludes_reposts_and_other_authors(self):
        own = candidate("own", "I live in Cebu City")
        other = candidate("other", "I live in Cebu City", author="did:other")
        client = Mock()
        client.app.bsky.feed.get_author_feed.return_value = {
            "feed": [{"post": own}, {"post": other}, {"post": own, "reason": {"type": "repost"}}]}
        ns = {"client": client, "get_attr": attr,
              "_call_with_rate_limit_retry": lambda fn, **kwargs: fn(**kwargs)}
        load_functions("bluesky.py", {"get_recent_post_texts"}, ns)
        self.assertEqual(ns["get_recent_post_texts"]("did:test", limit=5), ["I live in Cebu City"])

    def test_live_terminal_matches_complete_run_file(self):
        console = next(h for h in rules.logger.handlers
                       if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler))
        output = io.StringIO()
        with patch.object(console, "stream", output), patch.object(console, "level", logging.DEBUG):
            rules.logger.setLevel(logging.DEBUG)
            with rules.RetrievalRun(self.bio, self.recent) as run:
                run.query("sunog Cebu", 2)
                self.assertIn("SEARCH_QUERY", output.getvalue())
                self.assertIn("SEARCH_QUERY", run.log_path.read_text(encoding="utf-8"))
                original = "Sunog sa Cebu City\n  Please help: https://example.test/info ☀"
                run.evaluate(original, post_id="accepted", actor="did:test",
                             author_did="did:test", author_handle="test.bsky.social",
                             search_query="sunog Cebu")
                rejected_text = "Music in Cebu City\n  Details: https://example.test/music 🎵"
                run.evaluate(rejected_text, post_id="rejected", actor="did:rejected",
                             author_did="did:rejected", author_handle="rejected.bsky.social",
                             search_query="Cebu City")
                # Logs from provider helpers inherit this run and reach its file.
                rules.logger.warning("[RETRIEVAL_TRACE] PROVIDER_TEST")
            saved = run.log_path.read_text(encoding="utf-8")
        self.assertEqual(output.getvalue(), saved)
        for stage in ("START_RETRIEVAL", "SEARCH_QUERY", "DECISION=ACCEPT", "DECISION=REJECT",
                      "PROVIDER_TEST", "RETRIEVAL SUMMARY", "END_RETRIEVAL"):
            self.assertIn(stage, saved)
        self.assertIsNone(run._file_handler.stream)
        self.assertNotIn(run._file_handler, rules.logger.handlers)
        entry, = accepted_log_entries(run.log_path)
        self.assertEqual(entry["post_id"], "accepted")
        self.assertEqual(entry["author_did"], "did:test")
        self.assertEqual(entry["author_handle"], "test.bsky.social")
        self.assertEqual(entry["text"], original)
        self.assertEqual(entry["search_query"], "sunog Cebu")
        self.assertEqual(entry["reason"], "DISASTER_AND_VERIFIED_CEBU_LOCATION")
        self.assertEqual(entry["verification"], "EXPLICIT_CEBU")
        rejected_line, = [line for line in saved.splitlines() if "DECISION=REJECT " in line]
        rejected = json.loads(rejected_line.split("DECISION=REJECT ", 1)[1])
        self.assertEqual(rejected["post_id"], "rejected")
        self.assertEqual(rejected["author_did"], "did:rejected")
        self.assertEqual(rejected["author_handle"], "rejected.bsky.social")
        self.assertEqual(rejected["text"], rejected_text)
        self.assertEqual(rejected["search_query"], "Cebu City")
        self.assertEqual(rejected["reason"], "NO_DISASTER_MATCH")
        self.assertIsNone(rejected["verification"])
        for text in (
            "Status: ACCEPTED", "Status: DECLINED", "Post ID or URI: accepted",
            "Author DID: did:test", "Author handle: test.bsky.social",
            "Search query: sunog Cebu", "Full post text:\n" + original,
            "Full post text:\n" + rejected_text,
            "Decision reason: Disaster keywords, Cebu verification, and model relevance checks passed.",
            "Decision reason: No supported disaster keywords matched.",
            "Cebu verification method: Explicit Cebu mention in the post",
            "Cebu verification method: Not verified",
        ):
            self.assertIn(text, saved)

    def test_readable_decisions_are_visible_without_debug_logging(self):
        console = next(h for h in rules.logger.handlers
                       if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler))
        output = io.StringIO()
        with patch.object(console, "stream", output), patch.object(console, "level", logging.INFO):
            rules.logger.setLevel(logging.DEBUG)
            with rules.RetrievalRun(self.bio, self.recent) as run:
                run.query("baha")
                run.evaluate("Baha sa Cebu City", post_id="accepted", search_query="baha Cebu")
                run.evaluate("Music in Cebu City", post_id="declined", search_query="baha Cebu")
            saved = run.log_path.read_text(encoding="utf-8")
        for status in ("ACCEPTED", "DECLINED"):
            self.assertIn("Status: " + status, output.getvalue())
            self.assertIn("Status: " + status, saved)
            header = f"[RETRIEVAL POST] run={run.id}\nSearch query: baha\nStatus: {status}"
            self.assertIn(header, output.getvalue())
            self.assertIn(header, saved)
        self.assertIn("Author DID: Not available", output.getvalue())
        self.assertIn("Search query: baha Cebu", output.getvalue())
        self.assertNotIn("DISASTER_CHECK", output.getvalue())

    def test_failed_run_still_saves_summary_and_closes_file(self):
        rules.logger.setLevel(logging.DEBUG)
        run = rules.RetrievalRun(self.bio, self.recent)
        with self.assertRaises(RuntimeError):
            with run:
                run.query("sunog Cebu")
                raise RuntimeError("not recorded: private credential")
        saved = run.log_path.read_text(encoding="utf-8")
        self.assertIn("ERROR", saved)
        self.assertIn("RETRIEVAL SUMMARY", saved)
        self.assertIn("END_RETRIEVAL", saved)
        self.assertNotIn("private credential", saved)
        self.assertNotIn(run._file_handler, rules.logger.handlers)
        run.finish()  # Repeated cleanup must be harmless.

    def test_concurrent_run_files_do_not_mix(self):
        from threading import Barrier
        barrier = Barrier(2)
        rules.logger.setLevel(logging.DEBUG)

        def retrieve(marker):
            with rules.RetrievalRun(self.bio, self.recent) as run:
                barrier.wait(timeout=10)
                run.log(marker)
                rules.logger.warning("[RETRIEVAL_TRACE] provider_%s", marker)
                barrier.wait(timeout=10)
            return run.log_path

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(retrieve, "FIRST_ONLY")
            second = pool.submit(retrieve, "SECOND_ONLY")
            first_path, second_path = first.result(), second.result()
        self.assertNotEqual(first_path, second_path)
        first_log, second_log = first_path.read_text(), second_path.read_text()
        self.assertIn("FIRST_ONLY", first_log)
        self.assertNotIn("SECOND_ONLY", first_log)
        self.assertIn("SECOND_ONLY", second_log)
        self.assertNotIn("FIRST_ONLY", second_log)


if __name__ == "__main__":
    unittest.main()
