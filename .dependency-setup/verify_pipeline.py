import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, r"C:\AgapAI\AgapAI\agapai\api")
sys.path.insert(0, r"C:\AgapAI\AgapAI\agapai\api\tests")

from test_ner_pipeline import NerPipelineTests, post

# Reuse the isolated database setup, with the actual cached calamanCy model.
NerPipelineTests.setUpClass()
try:
    NerPipelineTests.posts.insert_many([
        post("at://smoke/loaded", "May baha sa Talisay."),
        post("at://smoke/outside", "May baha sa Talisay."),
    ])
    selection = {"post_ids": ["at://smoke/loaded"]}
    response = NerPipelineTests.api.post("/ner/process-all?limit=100", json=selection)
    assert response.status_code == 200, response.text
    assert response.json()["processed"] == 1, response.text
    response = NerPipelineTests.api.post("/ner/results", json=selection)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["total"] == 1, data
    row = data["rows"][0]
    assert row["ner_locations"], row
    assert row["ner_event_date"] == "2026-10-28", row
    assert row["ner_datetime_source"] == "fallback", row
    assert len(data["events"]) == 1, data
    assert data["events"][0]["posts"][0]["_id"] == row["_id"], data
    assert "ner_processed_at" not in NerPipelineTests.posts.find_one({"_id": "at://smoke/outside"})
    print("Real cached model: selected post processed; location, date/time, table and event verified.")
    print("Location:", row["ner_locations"][0]["matched_name"])
    print("Event date/time:", row["ner_event_date"], row["ner_event_time"])
finally:
    NerPipelineTests.tearDownClass()
