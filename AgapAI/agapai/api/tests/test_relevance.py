"""Gate scoring tests without loading models or contacting external services."""
import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from agapai_api import relevance


class RelevanceTests(unittest.TestCase):
    def classify(self, positive, negative):
        models = ModuleType("agapai_api.local_models")
        models.clean_for_classification = lambda text: text
        models.detect_source_language_code = lambda text: "ceb_Latn"
        models.translate_text = Mock(return_value="English translation")
        classifier = Mock(return_value={
            "labels": list(relevance.DISASTER_LABELS + relevance.NON_DISASTER_LABELS),
            "scores": [positive, 0.0, 0.0, negative, 0.0],
        })
        models.get_zero_shot_classifier = lambda: classifier
        with patch.dict(sys.modules, {models.__name__: models}):
            result = relevance.analyze_disaster_relevance("Cebuano post")
        models.translate_text.assert_called_once_with("Cebuano post", "ceb_Latn", "eng_Latn")
        return result

    def test_confident_literal_report_passes(self):
        self.assertTrue(self.classify(0.95, 0.05)["relevant"])

    def test_metaphor_and_uncertainty_fail(self):
        for positive in (0.1, 0.5, 0.59):
            self.assertFalse(self.classify(positive, 1 - positive)["relevant"])

    def test_invalid_score_raises(self):
        with self.assertRaises(ValueError):
            self.classify(float("nan"), 0.2)
