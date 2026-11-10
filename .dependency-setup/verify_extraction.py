import os
import sys
from datetime import datetime, timezone

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'C:\AgapAI\AgapAI\agapai\api')
sys.path.insert(0, r'C:\AgapAI\AgapAI\agapai\api\tests')
from test_ner_pipeline import NerPipelineTests, post
from agapai_api import ner_algorithm as ner
from agapai_api.geography import extract_location_name

samples = [
    'Cebu - Fire traps elderly woman in Cebu home, leading to tragic death #Cebu #News',
    "Guys there's a sunog malapit here sa Banilad UC school and the students are panicking we need fire responders ASAP please!!!",
    'Tabang po mga kapamilya, naay need og tambal ari sa Brgy. Pardo daghan mga sugatan tungods sa baha',
    'nasunog ang 20 na mga bahay dito sa brgy. san vicente cebu kaninang tanghali 12, wala ng natirang damit sa mga nasunogan. We need a wearable clothes for the mean time.',
]

NerPipelineTests.setUpClass()
try:
    rows = []
    for index, text in enumerate(samples):
        doc = post(f'sample-{index}', text, created_at='2026-10-06T08:09:33Z', location_name=extract_location_name(text))
        NerPipelineTests.posts.insert_one(doc)
        selection = {'post_ids': [doc['_id']]}
        response = NerPipelineTests.api.post('/ner/process-all', json=selection)
        assert response.status_code == 200, response.text
        response = NerPipelineTests.api.post('/ner/results', json=selection)
        assert response.status_code == 200, response.text
        row = response.json()['rows'][0]
        rows.append(row)
        print('Sample', index + 1, 'locations:', [loc['matched_name'] for loc in row['ner_locations']],
              'ambiguous:', [loc['location_text'] for loc in row['ner_ambiguous_locations']],
              'time:', row['ner_event_time'], flush=True)
    assert len(rows[0]['ner_locations']) == 1, rows[0]
    assert rows[0]['ner_locations'][0]['matched_name'] == 'City of Cebu', rows[0]
    assert rows[1]['ner_locations'] == [], rows[1]
    assert rows[1]['ner_ambiguous_locations'][0]['location_text'] == 'Banilad', rows[1]
    assert [loc['matched_name'] for loc in rows[2]['ner_locations']] == ['Pardo'], rows[2]
    assert rows[2]['ner_ambiguous_locations'] == [], rows[2]
    assert rows[3]['ner_event_time'] == '12:00:00 PM', rows[3]
    for text in ('Sunog sa Banilad', 'Naay sunog sa Banilad', 'May sunog sa Banilad'):
        result = ner.resolve_post_location(text, ground_truth_location_name='Sunog')
        assert not any(loc['matched_name'] == 'Sunog' for loc in result['locations']), (text, result)
    print('All four screenshot examples and additional incident/place variants passed with the cached model.', flush=True)
finally:
    NerPipelineTests.tearDownClass()
