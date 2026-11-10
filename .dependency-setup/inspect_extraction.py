import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'C:\AgapAI\AgapAI\agapai\api')
from agapai_api.ner_gazetteer_loader import _load_gazetteer

lookup = _load_gazetteer()
for key in ['pardo', 'banilad', 'sunog', 'san vicente']:
    print(key, json.dumps({name: entries for name, entries in lookup.items() if key in name}))
data = json.loads(Path(r'C:\AgapAI\AgapAI\agapai\api\agapai_api\cebu_gazetteer_final.json').read_text(encoding='utf-8'))
print('Landmarks:', data.get('landmarks'))
print('Pardo entries:', {name: value for name, value in data['barangays'].items() if 'pardo' in name.lower()})

from agapai_api import ner_algorithm as ner
from agapai_api.geography import extract_location_name

samples = [
    'Cebu - Fire traps elderly woman in Cebu home, leading to tragic death #Cebu #News',
    "Guys there's a sunog malapit here sa Banilad UC school and the students are panicking we need fire responders ASAP please!!!",
    'Tabang po mga kapamilya, naay need og tambal ari sa Brgy. Pardo daghan mga sugatan tungods sa baha',
    'nasunog ang 20 na mga bahay dito sa brgy. san vicente cebu kaninang tanghali 12, wala ng natirang damit sa mga nasunogan. We need a wearable clothes for the mean time.',
]
model = ner.get_nlp()
print('Model entity labels:', model.get_pipe('ner').labels)
for text in samples:
    print('\nTEXT', text)
    print('Entities', [(e.text, e.label_) for e in model(text).ents])
    location = extract_location_name(text)
    print('Retrieval location', location)
    print('Resolved', json.dumps(ner.resolve_post_location(text, ground_truth_location_name=location)))
    print('Time', ner.extract_explicit_datetime(text, datetime(2026, 10, 6, 8, 9, 33, tzinfo=timezone.utc)))
