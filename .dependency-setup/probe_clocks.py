from datetime import datetime
from dateparser import DateDataParser
from dateparser.search import search_dates

settings = {'RELATIVE_BASE': datetime(2026, 10, 6, 16, 9, 33), 'TIMEZONE': 'Asia/Manila', 'RETURN_AS_TIMEZONE_AWARE': True, 'RETURN_TIME_AS_PERIOD': True}
for text in ('12 PM', '9:45 p.m.', '14:30', '12 noon', 'noon', '13 PM', '25:99', '12', '20 na', 'sa', 'kahapon', 'yesterday 9 PM', '2026-10-05 at 15:30'):
    for lang in ('en', 'tl'):
        print(text, lang, DateDataParser(languages=[lang], settings=settings).get_date_data(text), flush=True)
        print('SEARCH', search_dates('Fire at '+text, languages=[lang], settings=settings), flush=True)
