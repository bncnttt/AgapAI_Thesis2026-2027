import os
import sys
from datetime import datetime

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, r'C:\AgapAI\AgapAI\agapai\api')
from agapai_api.ner_algorithm import get_nlp
from agapai_api.ner_gazetteer_loader import find_exact

samples = [
 'Donation drive para sa mga naapektuhan ng sunog malapit sa pasil, san vicente. We will give clothes pagkain and kumot today 4pm.',
 'tungod sa kusog na ulan kagabi bumaha ng mataas karon alas 10 ng umaga dito sa maghaway, talisay cebu. we need drinking water at pagkain, pwede rin clothes.',
 "Guys there's a sunog malapit here sa Banilad UC school and the students are panicking we need fire responders ASAP please!!!",
 'nasunog ang 20 na mga bahay dito sa brgy. san vicente cebu kaninang tanghali 12, wala ng natirang damit sa mga nasunogan. We need a wearable clothes for the mean time.',
]
nlp = get_nlp()
for text in samples:
    doc = nlp(text)
    print('TEXT', text, flush=True)
    print('ENTS', [(e.text,e.label_) for e in doc.ents], flush=True)
    print('TOKENS', [(t.text,t.pos_,t.dep_,t.head.text) for t in doc], flush=True)
for place in ['pasil', 'san vicente', 'maghaway', 'talisay']:
    print('PSGC', place, [(e['name'], e.get('parent_city_municipality'), e['psgc_code']) for e in find_exact(place)], flush=True)
if '--translate' in sys.argv:
    from agapai_api.local_models import translate_text, detect_source_language_code
    for text in samples[:2] + ['Bumaha kagabi.', 'Bumaha kanina.', 'Bumaha kahapon.', 'kaninang tanghali 12', 'alas 10 ng umaga']:
        language = detect_source_language_code(text)
        print('TRANSLATION', language, text, '=>', translate_text(text, language), flush=True)
