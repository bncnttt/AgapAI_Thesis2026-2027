from datetime import datetime
from zoneinfo import ZoneInfo
from dateparser import DateDataParser
from dateparser.search import search_dates

settings = {'RELATIVE_BASE': datetime(2026, 10, 6, 16, 9, 33, tzinfo=ZoneInfo('Asia/Manila')), 'TIMEZONE': 'Asia/Manila', 'RETURN_AS_TIMEZONE_AWARE': True, 'RETURN_TIME_AS_PERIOD': True, 'PREFER_DATES_FROM': 'past'}
samples = [
 'nasunog ang 20 na mga bahay dito sa brgy. san vicente cebu kaninang tanghali 12, wala ng natirang damit',
 'kaninang tanghali 12', '12 ng tanghali', 'ganina alas 12 sa udto',
 '3 ng hapon', '7 sa buntag', 'at noon', 'at 9:45 p.m.', '20 houses burned; 12 injured',
 'Baha sa Purok 12', 'Baha kahapon', 'Fire yesterday 9 PM', 'Fire on October 5, 2026 at 3:30 PM',
 'Fire on 2026-10-05 at 15:30', 'Tabang po mga kapamilya, naay need og tambal ari sa Brgy. Pardo daghan mga sugatan tungods sa baha',
]
for text in samples:
    print('\nTEXT', text, flush=True)
    for languages in (['tl', 'en'], None):
        matches = search_dates(text, languages=languages, settings=settings, add_detected_language=True)
        print(languages, matches, flush=True)
        for phrase, parsed, language in matches or []:
            info = DateDataParser(languages=[language], settings=settings).get_date_data(phrase)
            strict = DateDataParser(languages=[language], settings={**settings, 'STRICT_PARSING': True}).get_date_data(phrase)
            print('DATA', phrase, info, 'STRICT', strict, flush=True)
