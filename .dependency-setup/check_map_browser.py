import json
from pathlib import Path
from playwright.sync_api import sync_playwright

output = Path(r'C:\AgapAI\.dependency-setup')
posts = [{'_id': 'test-1', 'text': 'Flood in Maghaway', 'author_handle': 'test.bsky.social', 'disaster_type': 'flood', 'role': 'victim', 'created_at': '2026-10-29T02:00:00Z'}]
events = [{'event_id': 'event-0', 'lat': 10.27452, 'lon': 123.82078, 'place': 'Maghaway', 'disaster_type': 'flood', 'event_date': '2026-10-29', 'event_time': '10:00:00 AM', 'post_count': 1, 'posts': posts},
          {'event_id': 'event-1', 'lat': 10.296, 'lon': 123.89, 'place': 'Pasil', 'disaster_type': 'fire', 'event_date': '2026-10-29', 'event_time': '04:00:00 PM', 'post_count': 1, 'posts': [dict(posts[0], _id='test-2')]}]
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe', headless=True)
    page = browser.new_page(viewport={'width': 1400, 'height': 1000})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    def api(route):
        url = route.request.url
        if '/ner/process-all' in url:
            data = {'processed': 2, 'remaining': 0, 'batch_size': 100}
        elif '/ner/results' in url:
            data = {'rows': posts, 'events': events, 'total': 2}
        else:
            data = {'posts_collection': posts, 'users_collection': []}
        route.fulfill(status=200, content_type='application/json', body=json.dumps(data))
    page.route('http://127.0.0.1:8000/**', api)
    page.goto('http://127.0.0.1:5174', wait_until='networkidle')
    page.locator('input[type=date]').nth(0).fill('2026-10-01')
    page.locator('input[type=date]').nth(1).fill('2026-10-29')
    page.get_by_role('button', name='Load Data', exact=True).click()
    page.wait_for_function("document.querySelectorAll('.agapai-disaster-marker').length === 2")
    page.get_by_role('button', name='Map View', exact=True).click()
    page.locator('.leaflet-container').wait_for(state='visible')
    page.wait_for_timeout(2000)
    for width, height in [(1400, 1000), (900, 900)]:
        page.set_viewport_size({'width': width, 'height': height})
        page.wait_for_timeout(800)
        bounds = page.locator('.leaflet-container').bounding_box()
        markers = page.locator('.agapai-disaster-marker').all()
        for marker in markers:
            box = marker.bounding_box()
            assert box and bounds['x'] <= box['x'] and box['x'] + box['width'] <= bounds['x'] + bounds['width'], (bounds, box)
            assert bounds['y'] <= box['y'] and box['y'] + box['height'] <= bounds['y'] + bounds['height'], (bounds, box)
        print('Visible markers after tab/resize', width, len(markers), flush=True)
    page.get_by_role('button', name='Spatiotemporal Info', exact=True).click()
    page.get_by_role('button', name='Map View', exact=True).click()
    page.wait_for_timeout(800)
    page.locator('.agapai-disaster-marker').first.click()
    page.locator('.leaflet-popup-content').wait_for(state='visible')
    print('Tiles', page.locator('.leaflet-tile').evaluate_all('(tiles) => tiles.map(t => ({loaded: t.complete && t.naturalWidth > 0, url: t.src}))'), flush=True)
    print('JS errors', errors, flush=True)
    assert not errors
    page.screenshot(path=str(output / 'map-verified.png'), full_page=True)
    browser.close()
