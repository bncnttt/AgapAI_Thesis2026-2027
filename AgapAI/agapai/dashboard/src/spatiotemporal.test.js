import assert from "node:assert/strict";
import { test } from "node:test";
import { loadSpatiotemporalData, locationDisplayItems } from "./spatiotemporal.js";

function respond(data, ok = true) {
  return { ok, json: async () => data };
}

test("repeated place IDs display once while different barangays stay distinct", () => {
  const location = { psgc_code: "cebu-banilad", matched_name: "Banilad", breadcrumb: ["Region", "Cebu", "City of Cebu", "Banilad"], confidence: "HIGH" };
  const other = { ...location, psgc_code: "mandaue-banilad", breadcrumb: ["Region", "Cebu", "City of Mandaue", "Banilad"] };
  const items = locationDisplayItems({ ner_locations: [location, location, other] });
  assert.deepEqual(items.map((item) => item.name), ["Banilad, City of Cebu", "Banilad, City of Mandaue"]);
  assert.deepEqual(items.map((item) => item.confidence), ["HIGH", "HIGH"]);
});

test("recognized ambiguous names remain visible without pretending to have a resolved location", () => {
  const items = locationDisplayItems({ ner_ambiguous_locations: [
    { location_text: "Banilad", candidates: [{ psgc_code: "cebu", name: "Banilad" }, { psgc_code: "mandaue", name: "Banilad" }] },
    { location_text: "Brgy.", candidates: [] },
  ] });
  assert.deepEqual(items, [{ name: "Banilad (location unresolved)", confidence: "Unresolved" }]);
});

test("processes all batches for the loaded IDs before fetching shared results", async (t) => {
  const posts = Array.from({ length: 101 }, (_, i) => ({ _id: `at://loaded/${i}` }));
  const results = { rows: posts, events: [{ event_id: "event-0" }], total: 101 };
  const responses = [
    respond({ processed: 100, remaining: 1 }), respond({ processed: 1, remaining: 0 }), respond(results),
  ];
  const requests = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    requests.push({ url, options });
    return responses.shift();
  });
  assert.deepEqual(await loadSpatiotemporalData(posts), results);
  assert.equal(requests.length, 3);
  assert.ok(requests[0].url.endsWith("process-all?limit=100"));
  assert.ok(requests[2].url.endsWith("results"));
  for (const { options } of requests) {
    assert.equal(options.method, "POST");
    assert.deepEqual(JSON.parse(options.body).post_ids, posts.map((post) => post._id));
  }
});

test("empty retrieval clears results without a global request", async (t) => {
  const fetch = t.mock.method(globalThis, "fetch");
  assert.deepEqual(await loadSpatiotemporalData([]), { rows: [], events: [], total: 0 });
  assert.equal(fetch.mock.callCount(), 0);
});

test("a new selection replaces previous IDs and cached posts need no extra batches", async (t) => {
  const requests = [];
  t.mock.method(globalThis, "fetch", async (url, options) => {
    const ids = JSON.parse(options.body).post_ids;
    requests.push(ids);
    return respond(url.endsWith("results") ? { rows: ids, events: [] } : { processed: 0, remaining: 0 });
  });
  await loadSpatiotemporalData([{ _id: "first" }, { _id: "first" }]);
  const data = await loadSpatiotemporalData([{ _id: "second" }]);
  assert.deepEqual(requests, [["first"], ["first"], ["second"], ["second"]]);
  assert.deepEqual(data.rows, ["second"]);
});

test("processing errors do not fetch stale results", async (t) => {
  const fetch = t.mock.method(globalThis, "fetch", async () => respond({ detail: "Extraction failed" }, false));
  await assert.rejects(loadSpatiotemporalData([{ _id: "loaded" }]), /Extraction failed/);
  assert.equal(fetch.mock.callCount(), 1);
});

test("cancellation prevents subsequent batches and results", async (t) => {
  const controller = new AbortController();
  const fetch = t.mock.method(globalThis, "fetch", async () => {
    controller.abort();
    return respond({ processed: 100, remaining: 1 });
  });
  await assert.rejects(loadSpatiotemporalData([{ _id: "loaded" }], { signal: controller.signal }), { name: "AbortError" });
  assert.equal(fetch.mock.callCount(), 1);
});

test("a stalled batch fails instead of looping forever", async (t) => {
  const fetch = t.mock.method(globalThis, "fetch", async () => respond({ processed: 0, remaining: 1 }));
  await assert.rejects(loadSpatiotemporalData([{ _id: "loaded" }]), /stopped before all loaded posts/);
  assert.equal(fetch.mock.callCount(), 1);
});
