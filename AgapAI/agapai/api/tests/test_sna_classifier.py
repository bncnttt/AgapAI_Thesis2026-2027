"""SNA integration checks with isolated classifier and database doubles."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


clients = ModuleType("agapai_api.clients")
for name in ("authenticate_bluesky", "client", "is_bluesky_authenticated", "posts_col", "users_col"):
    setattr(clients, name, Mock())
clients.mongo_connected = True
spec = importlib.util.spec_from_file_location(
    "isolated_sna", Path(__file__).resolve().parents[1] / "agapai_api" / "sna_router.py")
sna = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"agapai_api.clients": clients}):
    spec.loader.exec_module(sna)


def prediction(role="Victim", intent="need", needs=None, offers=None):
    return {"role": role, "intent": intent, "needs": needs or {},
            "offers": offers or {}, "urgency": "Very High"}


class SnaClassifierTests(unittest.TestCase):
    def setUp(self):
        sna._analysis_cache.clear()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.logs = Path(temporary.name)
        env = patch.dict(sna.os.environ, {"AGAPAI_SNA_LOG_DIR": str(self.logs)})
        env.start()
        self.addCleanup(env.stop)

    def test_sna_log_saves_result_and_closes_file(self):
        handlers = list(sna.logger.handlers)

        @sna.log_sna_analysis("graph")
        def analysis():
            sna.logger.info("Candidate checked: volunteer.example")
            return {"target": {"handle": "victim.example", "post_text": "Need clothes"},
                    "matches": [{"handle": "volunteer.example", "hop": 1}]}

        result = analysis()
        path, = self.logs.glob("sna_graph_*.log")
        saved = path.read_text(encoding="utf-8")
        for marker in ("START graph", "Candidate checked", "victim.example", "Need clothes",
                       "volunteer.example", "END graph"):
            self.assertIn(marker, saved)
        self.assertEqual(result["matches"][0]["hop"], 1)
        self.assertEqual(sna.logger.handlers, handlers)

    def test_failed_analysis_is_logged_and_exception_preserved(self):
        @sna.log_sna_analysis("graph")
        def analysis():
            raise ValueError("private provider response")

        with self.assertRaises(ValueError):
            analysis()
        path, = self.logs.glob("*.log")
        saved = path.read_text(encoding="utf-8")
        self.assertIn("ANALYSIS FAILED: ValueError", saved)
        self.assertIn("END graph", saved)
        self.assertNotIn("private provider response", saved)

    def test_concurrent_analysis_files_do_not_mix(self):
        barrier = Barrier(2)

        @sna.log_sna_analysis("graph")
        def analysis(marker):
            barrier.wait(timeout=10)
            sna.logger.info("Candidate %s", marker)
            return {"target": {"handle": marker}}

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(analysis, ("FIRST_ONLY", "SECOND_ONLY")))
        saved = [path.read_text(encoding="utf-8") for path in self.logs.glob("*.log")]
        self.assertEqual(len(saved), 2)
        for text in saved:
            self.assertNotEqual("FIRST_ONLY" in text, "SECOND_ONLY" in text)

    def test_saved_classifier_wins_over_stale_fields(self):
        post = {"classifier_prediction": prediction(role="General Info", intent="none"),
                "role": "Victim", "needs": ["Food"], "donations": ["Water"]}
        self.assertIsNone(sna.infer_post_role(post))
        self.assertFalse(sna.is_help_seeking_victim(post))
        self.assertEqual(sna.infer_needed_items(post), [])
        self.assertEqual(sna.infer_offered_items(post), [])

    def test_victim_requires_classifier_role_intent_and_resources(self):
        self.assertTrue(sna.is_help_seeking_victim({
            "classifier_prediction": prediction(needs={"Food": .9})}))
        self.assertFalse(sna.is_help_seeking_victim({
            "classifier_prediction": prediction(role="Volunteer", needs={"Food": .9})}))
        self.assertFalse(sna.is_help_seeking_victim({
            "classifier_prediction": prediction()}))

    def test_fallback_uses_existing_classify_module_and_caches(self):
        module = ModuleType("agapai_api.classify")
        module.classify = Mock(return_value=prediction(needs={"Water": .8}))
        with patch.dict(sys.modules, {"agapai_api.classify": module}):
            self.assertEqual(sna.infer_needed_items({"text": "Need water"}), ["water"])
            self.assertTrue(sna.is_help_seeking_victim({"text": "Need water"}))
        module.classify.assert_called_once_with("Need water")

    def test_saved_predictions_are_not_reclassified_by_batch_warming(self):
        with patch.object(sna, "analyze_post_text") as analyze:
            sna.warm_analysis_cache([{"text": "saved", "classifier_prediction": prediction()}])
            analyze.assert_not_called()
        self.assertEqual(sna._analysis_cache, {})

    def test_queue_exposes_classifier_resources_and_very_high_urgency(self):
        collection = Mock()
        collection.count_documents.return_value = 1
        collection.find.return_value.sort.return_value = [{
            "_id": "post", "author_did": "did:plc:victim",
            "classifier_prediction": prediction(needs={"Food": .9, "Water": .8})}]
        with patch.object(sna, "posts_col", collection):
            incident, = sna.get_sna_incidents()["incidents"]
        self.assertEqual(incident["needed_items"], ["Food", "Water"])
        self.assertEqual(incident["urgency_level"], "Very High")

    def test_offer_matching_uses_classifier_resources_not_search_query(self):
        collection = Mock()
        collection.find.return_value.sort.return_value.limit.return_value = [{
            "_id": "offer", "author_did": "did:plc:donor",
            "classifier_prediction": prediction(role="Volunteer", intent="offer", offers={"Water": .8})}]
        with patch.object(sna, "posts_col", collection):
            match, = sna.get_item_matched_counterparts("Victim", ["water"])
        self.assertEqual(match["matched_items"], ["Water"])
        self.assertEqual(match["role"], "Volunteer")
        self.assertEqual(sna._stored_items({"search_query": "baha Cebu"}, sna.NEED_KEYS), [])

    def test_victim_with_conflicting_offer_is_not_a_volunteer_match(self):
        post = {
            "_id": "victim-clothes", "author_did": "did:plc:victim",
            "text": "Nasunugan ang buong purok namin sa Barili, Cebu bandang 4:00 AM. "
                    "Walang natira sa amin kundi ang aming mga suot na damit.",
            "classifier_prediction": prediction(role="Victim", intent="offer",
                                                offers={"Clothing and Bedding": .871}),
        }
        self.assertFalse(sna.is_offering_volunteer(post))
        collection = Mock()
        collection.find.return_value.sort.return_value.limit.return_value = [post]
        with patch.object(sna, "posts_col", collection):
            self.assertEqual(sna.get_item_matched_counterparts("Victim", ["clothing and bedding"]), [])

    def test_general_information_offer_is_not_labelled_as_volunteer(self):
        self.assertFalse(sna.is_offering_volunteer({"classifier_prediction":
            prediction(role="General Info", intent="offer", offers={"Water": .8})}))


if __name__ == "__main__":
    unittest.main()
