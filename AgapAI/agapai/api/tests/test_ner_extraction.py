"""Model predictions are fixtures; PSGC, source cascade, dates and DBSCAN run."""
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import spacy

from agapai_api import ner_algorithm as ner
from agapai_api.ner_gazetteer_loader import _load_gazetteer, find_exact
from agapai_api.ner_event_clustering import cluster_posts_into_events, build_event_summary, MAX_TIME_HOURS


class LocationExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = spacy.blank("tl")

    def doc(self, text, entities=(), proper_nouns=()):
        doc = self.tokenizer(text)
        spans = []
        for phrase, label in entities:
            offset = 0
            while (offset := text.find(phrase, offset)) >= 0:
                spans.append(doc.char_span(offset, offset + len(phrase), label=label))
                offset += len(phrase)
        doc.ents = spans
        for token in doc:
            if token.text in proper_nouns:
                token.pos_ = "PROPN"
        return doc

    def test_repeated_mentions_have_one_identity_and_bare_cebu_is_province(self):
        text = "Cebu - Fire in Cebu home #Cebu"
        result = ner.resolve_post_location(text, doc=self.doc(text, [("Cebu", "LOC")]))
        self.assertEqual(len(result["locations"]), 1)
        self.assertEqual(result["locations"][0]["matched_name"], "Cebu")
        self.assertEqual(result["locations"][0]["psgc_code"], find_exact("Cebu")[0]["psgc_code"])
        self.assertEqual(result["ambiguous_locations"], [])
        self.assertEqual(find_exact("Cebu")[0]["type"], "Province")
        self.assertEqual(find_exact("Cebu City")[0]["name"], "City of Cebu")

    def test_proper_noun_inside_organization_is_validated_without_scanning_incident_nouns(self):
        text = "There is a sunog here sa Banilad UC school"
        doc = self.doc(text, [("Banilad UC school", "ORG")], ("Banilad", "UC"))
        result = ner.resolve_post_location(text, doc=doc)
        self.assertEqual(result["locations"], [])
        self.assertEqual(result["ambiguous_locations"][0]["location_text"], "Banilad")
        self.assertEqual(result["ambiguous_locations"][0]["candidate_count"], 2)
        # No model evidence means no whole-post dictionary fallback.
        self.assertEqual(ner.resolve_post_location(text, doc=self.doc(text))["locations"], [])

    def test_nominal_modifier_recovers_missed_province_in_headline(self):
        text = "Cebu archbishop warns of values crisis, urges conscience formation"
        doc = self.doc(text)
        doc[0].pos_, doc[0].dep_ = "NOUN", "nmod"
        result = ner.resolve_post_location(text, doc=doc)
        self.assertEqual(result["locations"][0]["matched_name"], "Cebu")
        self.assertEqual(result["locations"][0]["confidence"], "HIGH")

    def test_partial_model_entity_recovers_proper_name_and_excludes_following_prose(self):
        text = "Tabang ari sa Brgy. Pardo daghan mga sugatan"
        doc = self.doc(text, [("Brgy.", "LOC")], ("Pardo",))
        result = ner.resolve_post_location(text, doc=doc)
        self.assertEqual([loc["matched_name"] for loc in result["locations"]], ["Pardo"])
        self.assertEqual(result["ambiguous_locations"], [])

    def test_longest_names_and_homonyms_use_model_evidence(self):
        for phrase, name in (("Brgy. Basak Pardo", "Basak Pardo"), ("Barangay Sunog, Balamban", "Sunog")):
            result = ner.resolve_post_location(phrase, doc=self.doc(phrase, [(phrase, "LOC")]))
            self.assertEqual(result["locations"][0]["matched_name"], name)

    def test_unknown_and_out_of_file_entities_never_get_coordinates(self):
        for name in ("Manila", "Pardohan", "Inventedplace", "Brgy."):
            result = ner.resolve_post_location(name, doc=self.doc(name, [(name, "LOC")]))
            self.assertEqual(result["locations"], [])
            self.assertEqual(result["ambiguous_locations"], [])

    def test_same_name_in_different_cities_keeps_distinct_psgc_ids(self):
        text = "Banilad, Cebu City and Banilad, Mandaue City"
        doc = self.doc(text, [("Banilad, Cebu City", "LOC"), ("Banilad, Mandaue City", "LOC")])
        result = ner.resolve_post_location(text, doc=doc)
        self.assertEqual(len({loc["psgc_code"] for loc in result["locations"]}), 2)
        self.assertEqual(result["ambiguous_locations"], [])

    def test_bare_province_does_not_invent_barangay_parent(self):
        text = "san vicente cebu"
        result = ner.resolve_post_location(text, doc=self.doc(text, [(text, "LOC")]))
        self.assertEqual(result["locations"], [])
        self.assertEqual(result["ambiguous_locations"][0]["candidate_count"], 4)

    def test_model_candidates_across_gazetteer_resolve_to_file_ids(self):
        entries = {item["psgc_code"]: item for values in _load_gazetteer().values() for item in values if item["type"] == "Barangay"}
        for entry in list(entries.values())[::75]:
            text = f"{entry['name']}, {entry['parent_city_municipality']}"
            with self.subTest(text=text):
                result = ner.resolve_post_location(text, doc=self.doc(text, [(text, "LOC")]))
                self.assertIn(entry["psgc_code"], {loc["psgc_code"] for loc in result["locations"]})

    def test_confidence_follows_post_then_bio_then_previous_ten(self):
        def predict(text):
            return self.doc(text, [(text, "LOC")]) if find_exact(text) else self.doc(text)
        with patch.object(ner, "get_nlp", return_value=predict):
            cases = [
                ("Pardo", "Talisay", ["Mandaue"], "Pardo", "HIGH", "post_text"),
                ("Help", "Pardo", ["Talisay"], "Pardo", "MEDIUM", "author_bio"),
                ("Help", "Volunteer", ["Pardo"], "Pardo", "LOW", "author_previous_post"),
            ]
            for post, bio, history, name, confidence, source in cases:
                result = ner.resolve_post_location(post, bio, history)
                location = result["locations"][0]
                self.assertEqual((location["matched_name"], location["confidence"], location["confidence_source"]), (name, confidence, source))
            result = ner.resolve_post_location("Help", "Volunteer", ["No place"] * 10 + ["Pardo"])
            self.assertEqual(result["locations"], [])

    def test_post_location_stops_bio_and_history_even_when_ambiguous(self):
        with patch.object(ner, "get_nlp", side_effect=lambda: lambda text: self.doc(text, [(text, "LOC")])):
            result = ner.resolve_post_location("Banilad", "Mandaue City", ["Pardo"])
        self.assertEqual(result["locations"], [])
        self.assertEqual(result["ambiguous_locations"][0]["location_text"], "Banilad")

    def test_lowercase_dependency_candidates_preserve_post_locations(self):
        text = "malapit sa pasil, san vicente"
        doc = self.doc(text)
        for token in doc:
            if token.text == "pasil":
                token.pos_, token.dep_ = "NOUN", "obl"
            elif token.text == "san":
                token.pos_, token.dep_ = "NOUN", "list"
            elif token.text == "vicente":
                token.pos_, token.dep_ = "NOUN", "flat"
        result = ner.resolve_post_location(text, author_bio="Sibonga", previous_post_texts=["Naga"], doc=doc)
        self.assertEqual(result["locations"], [])
        self.assertEqual({item["location_text"] for item in result["ambiguous_locations"]}, {"pasil", "san vicente"})

    def test_lowercase_address_can_have_a_mistagged_verb(self):
        text = "dito sa maghaway, talisay cebu"
        doc = self.doc(text)
        for token in doc:
            if token.text == "maghaway":
                token.pos_, token.dep_ = "VERB", "obl"
            elif token.text in ("talisay", "cebu"):
                token.pos_, token.dep_ = "NOUN", "compound"
        result = ner.resolve_post_location(text, author_bio="Naga", doc=doc)
        self.assertEqual([loc["matched_name"] for loc in result["locations"]], ["Maghaway"])
        self.assertEqual(result["locations"][0]["confidence"], "HIGH")
        self.assertEqual(result["locations"][0]["breadcrumb"][-2], "City of Talisay")

    def test_bio_location_stops_history_even_when_ambiguous(self):
        with patch.object(ner, "get_nlp", return_value=lambda text: self.doc(text, [(text, "LOC")]) if find_exact(text) else self.doc(text)):
            result = ner.resolve_post_location("Help", "Banilad", ["Pardo"])
        self.assertEqual(result["locations"], [])
        self.assertEqual(result["ambiguous_locations"][0]["location_text"], "Banilad")


class TemporalExtractionTests(unittest.TestCase):
    reference = datetime(2026, 10, 6, 8, 9, 33, tzinfo=timezone.utc)

    def setUp(self):
        translator = patch.object(ner, "_translate_temporal_text", return_value=None)
        self.translate = translator.start()
        self.addCleanup(translator.stop)

    def test_library_supported_clocks(self):
        for phrase, expected in (("12 PM", "12:00:00 PM"), ("9:45 p.m.", "09:45:00 PM"), ("14:30", "02:30:00 PM"), ("noon", "12:00:00 PM"), ("12 AM", "12:00:00 AM")):
            with self.subTest(phrase=phrase):
                result = ner.extract_explicit_datetime("Fire at " + phrase, self.reference)
                self.assertEqual(result["event_time"], expected)
                self.assertEqual(result["datetime_source"], "explicit")

    def test_relative_dates_use_post_timestamp_in_philippine_time(self):
        reference = datetime(2026, 10, 5, 18, 30, tzinfo=timezone.utc)
        for text in ("Baha kahapon", "Fire yesterday"):
            result = ner.extract_explicit_datetime(text, reference)
            self.assertEqual(result["event_date"], "2026-10-05")
            self.assertEqual(result["event_time"], "02:30:00 AM")

    def test_counts_and_unsupported_expressions_do_not_become_invented_times(self):
        for text in ("20 houses burned; 12 injured", "Baha sa Purok 12", "We need 12 shirts", "Fire at 25:99", "Fire at 13 PM", "kaninang tanghali 12"):
            with self.subTest(text=text):
                result = ner.extract_explicit_datetime(text, self.reference)
                self.assertEqual(result["datetime_source"], "fallback")
                self.assertEqual(result["event_datetime"], self.reference)

    def test_date_and_time_are_combined(self):
        for text in ("Fire on October 5, 2026 at 3:30 PM", "Fire on 2026-10-05 at 15:30"):
            result = ner.extract_explicit_datetime(text, self.reference)
            self.assertEqual(result["event_date"], "2026-10-05")
            self.assertEqual(result["event_time"], "03:30:00 PM")

    def test_fallback_is_labeled_and_timezone_consistent(self):
        reference = datetime(2026, 10, 5, 18, 30, tzinfo=timezone.utc)
        result = ner.extract_explicit_datetime("Baha sa Pardo", reference)
        self.assertEqual((result["event_date"], result["event_time"], result["source"]), ("2026-10-06", "02:30:00 AM", "fallback"))
        self.assertIsNone(ner.extract_explicit_datetime("Help")["event_datetime"])

    def test_relative_night_uses_posted_clock_when_phrase_has_no_clock(self):
        result = ner.extract_explicit_datetime("Fire last night", self.reference)
        self.assertEqual(result["event_date"], "2026-10-05")
        self.assertEqual(result["event_time"], "04:09:33 PM")
        self.assertEqual(result["source"], "vague_date")
        self.assertEqual(result["temporal_resolution"]["type"], "datetimerange")

    def test_relative_stage_uses_trained_translation_output(self):
        self.translate.return_value = "It was late last night."
        result = ner.extract_explicit_datetime("Bumaha kagabi.", self.reference)
        self.assertEqual(result["event_date"], "2026-10-05")
        self.assertEqual(result["event_time"], "04:09:33 PM")
        self.assertEqual(result["translated_phrase"], "last night")

    def test_explicit_clock_precedes_other_relative_mentions(self):
        self.translate.return_value = "Because of the heavy rain last night, we got up at 10am here in the morning."
        result = ner.extract_explicit_datetime("kagabi bumaha karon alas 10 ng umaga", self.reference)
        self.assertEqual(result["event_time"], "10:00:00 AM")
        self.assertEqual(result["source"], "explicit")
        result = ner.extract_explicit_datetime("Rain last night. Donations today 4pm.", self.reference)
        self.assertEqual(result["event_date"], "2026-10-06")
        self.assertEqual(result["event_time"], "04:00:00 PM")


class EventClusteringTests(unittest.TestCase):
    def test_distance_time_and_disaster_class_control_clusters(self):
        entry = find_exact("Pardo")[0]
        base = {"lat": entry["latitude"], "lon": entry["longitude"], "disaster_type": "fire", "created_at": TemporalExtractionTests.reference,
                "location_text": entry["name"], "location_confidence": "HIGH", "datetime_source": "fallback"}
        posts = [dict(base, _id="a"), dict(base, _id="b"), dict(base, _id="far", lat=base["lat"] + 1),
                 dict(base, _id="late", created_at=base["created_at"] + timedelta(hours=MAX_TIME_HOURS + 1)), dict(base, _id="other", disaster_type="flood")]
        clusters = cluster_posts_into_events(posts)
        self.assertEqual(sorted(len(cluster) for cluster in clusters), [1, 1, 1, 2])
        cluster = next(cluster for cluster in clusters if len(cluster) == 2)
        summary = build_event_summary(cluster)
        self.assertEqual(summary["location"]["place"], "Pardo")
        self.assertEqual(summary["event_datetime"], base["created_at"])


if __name__ == "__main__":
    unittest.main()
