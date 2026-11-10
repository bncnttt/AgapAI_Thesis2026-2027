import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, r'C:\AgapAI\AgapAI\agapai\api')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from dateparser import DateDataParser
from dateparser.search import search_dates

settings = {'RELATIVE_BASE': datetime(2026, 10, 6, 16, 9, 33, tzinfo=ZoneInfo('Asia/Manila')), 'TIMEZONE': 'Asia/Manila', 'RETURN_AS_TIMEZONE_AWARE': True, 'RETURN_TIME_AS_PERIOD': True, 'PREFER_DATES_FROM': 'past'}
samples = ['Fire at 12 noon', 'Fire at noon', 'Fire at 12 PM', 'Fire at 9:45 p.m.', 'Fire at 14:30', 'Baha kahapon', 'Fire at 25:99', 'Fire at 13 PM', '20 houses burned; 12 injured', 'nasunog ang 20 na mga bahay kaninang tanghali 12']
for text in samples:
    print('TEXT', text, flush=True)
    for language in ('en', 'tl'):
        matches = search_dates(text, languages=[language], settings=settings, strategy='ngram')
        for phrase, parsed in matches or []:
            print(language, phrase, DateDataParser(languages=[language], settings=settings).get_date_data(phrase), flush=True)

if '--translate' in sys.argv:
    from agapai_api.local_models import translate_text, detect_source_language_code
    for text in ('san vicente cebu kaninang tanghali 12, wala ng natirang damit sa mga nasunogan.', 'kaninang tanghali 12', 'ganina alas 12 sa udto'):
        translated = translate_text(text, detect_source_language_code(text))
        print('TRANSLATED', text, translated, flush=True)
        print('PARSED', search_dates(translated, languages=['en'], settings=settings, strategy='ngram', add_detected_language=True), flush=True)
