import os
import sys
from datetime import datetime

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, r'C:\AgapAI\AgapAI\agapai\api')
from recognizers_date_time import recognize_datetime, Culture

reference = datetime(2026, 10, 29, 16, 30)
samples = ['Fire last night', 'Fire earlier', 'Fire earlier today', 'Fire a while ago', 'Fire yesterday', 'Fire at noon',
 'Because of the heavy rain last night, we got up at 10am here in the morning, we need drinking water and food, we can also get clothes.',
 'Fire that afternoon 12', 'Fire 12 noon today', 'Today 4pm', 'Fire last night at 10 AM', '20 houses burned; 12 injured', 'Baha sa Purok 12', 'kanina', 'kagabi', 'kahapon']
for text in samples:
    print(text, [(r.text, r.type_name, r.resolution) for r in recognize_datetime(text, Culture.English, reference=reference)], flush=True)
if '--translate' in sys.argv:
    from agapai_api.local_models import translate_text
    for text in ['Bumaha kanina.', 'kanina', 'kagabi', 'kahapon', 'kaninang tanghali 12', 'Baha kaninang tanghali 12.', 'tungod sa kusog na ulan kagabi bumaha ng mataas karon alas 10 ng umaga dito sa maghaway, talisay cebu.']:
        english = translate_text(text, 'tgl_Latn')
        print('TRANSLATE', text, '=>', english, flush=True)
        print('RESULT', [(r.text, r.type_name, r.resolution) for r in recognize_datetime(english, Culture.English, reference=reference)], flush=True)
