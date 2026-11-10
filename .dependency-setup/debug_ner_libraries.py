import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, r'C:\AgapAI\AgapAI\agapai\api')
from agapai_api.ner_algorithm import get_nlp
from dateparser import DateDataParser
from dateparser.search import search_dates

settings = {'RELATIVE_BASE': datetime(2026, 10, 6, 16, 9, 33, tzinfo=ZoneInfo('Asia/Manila')), 'TIMEZONE': 'Asia/Manila', 'RETURN_AS_TIMEZONE_AWARE': True, 'RETURN_TIME_AS_PERIOD': True, 'PREFER_DATES_FROM': 'past'}
for preference in ('past', 'current_period'):
    settings['PREFER_DATES_FROM'] = preference
    print(preference, search_dates('Fire at 12 PM', languages=['en'], settings=settings), DateDataParser(languages=['en'], settings=settings).get_date_data('12 PM'), flush=True)
nlp = get_nlp()
for text in ('Cebu archbishop warns of values crisis, urges conscience formation', 'Sunog sa Banilad', 'May sunog sa Banilad', 'Cebu', 'Banilad', 'Brgy. san vicente cebu'):
    doc = nlp(text)
    print(text, [(t.text, t.pos_, t.tag_, t.dep_, t.head.text) for t in doc], flush=True)
    print('NOUN CHUNKS', [(s.text, s.start_char) for s in doc.noun_chunks], flush=True)
