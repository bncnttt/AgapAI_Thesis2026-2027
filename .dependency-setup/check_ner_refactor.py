import os
import sys
from datetime import datetime, timezone

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'C:\AgapAI\AgapAI\agapai\api')
from agapai_api.ner_algorithm import get_nlp, resolve_post_location, extract_explicit_datetime

samples = [
 'Cebu archbishop warns of values crisis, urges conscience formation',
 'Cebu - Fire traps elderly woman in Cebu home, leading to tragic death #Cebu #News',
 "Guys there's a sunog malapit here sa Banilad UC school and the students are panicking we need fire responders ASAP please!!!",
 'Tabang po mga kapamilya, naay need og tambal ari sa Brgy. Pardo daghan mga sugatan tungods sa baha',
 'May baha sa Barangay Sunog, Balamban',
 'Flood in Banilad, Cebu City and Banilad, Mandaue City',
 'Flood in Manila',
]
nlp = get_nlp()
for text in samples:
    doc = nlp(text)
    print(text, [(e.text,e.label_) for e in doc.ents], flush=True)
    print(resolve_post_location(text, doc=doc), flush=True)
reference = datetime(2026, 10, 6, 8, 9, 33, tzinfo=timezone.utc)
for text in ('Fire at 12 PM', 'Fire at 9:45 p.m.', 'Fire at noon', 'Baha kahapon', 'Fire yesterday 9 PM', '20 houses burned; 12 injured', 'Fire on October 5, 2026 at 3:30 PM', 'kaninang tanghali 12'):
    print(text, extract_explicit_datetime(text, reference), flush=True)
