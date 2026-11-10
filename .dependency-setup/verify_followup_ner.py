import ast
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, r'C:\AgapAI\AgapAI\agapai\api')
from agapai_api.ner_algorithm import resolve_post_location, extract_explicit_datetime

tree = ast.parse(Path(r'C:\AgapAI\.dependency-setup\probe_followup_ner.py').read_text())
samples = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'samples' for t in node.targets))
first = resolve_post_location(samples[0], author_bio='Sibonga', previous_post_texts=['Naga'])
assert first['locations'] == [], first
assert {item['location_text'].casefold() for item in first['ambiguous_locations']} == {'pasil', 'san vicente'}, first
print('First screenshot: only Pasil and San Vicente, no author fallback.', flush=True)
second = resolve_post_location(samples[1], author_bio='Naga', previous_post_texts=['Sibonga'])
assert [loc['matched_name'] for loc in second['locations']] == ['Maghaway'], second
assert second['locations'][0]['breadcrumb'][-2] == 'City of Talisay', second
assert second['locations'][0]['confidence'] == 'HIGH', second
print('Second screenshot: Maghaway, City of Talisay, HIGH.', flush=True)
reference = datetime(2026, 10, 29, 8, 30, tzinfo=timezone.utc)
for text, expected in [(samples[0], '04:00:00 PM'), (samples[1], '10:00:00 AM')]:
    result = extract_explicit_datetime(text, reference)
    assert result['event_time'] == expected, result
    print('Screenshot temporal result:', result, flush=True)
for text in ('Bumaha kagabi.', 'Baha kahapon.', 'Fire last night', 'Bumaha kanina.', 'We need clothes'):
    result = extract_explicit_datetime(text, reference)
    print(text, result, flush=True)
