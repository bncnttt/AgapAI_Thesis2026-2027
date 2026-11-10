const API_BASE = "http://127.0.0.1:8000/ner";

export function locationDisplayItems(row) {
  const items = new Map();
  for (const loc of row.ner_locations || []) {
    const [, , municipality, barangay] = loc.breadcrumb || [];
    const name = barangay && municipality ? `${barangay}, ${municipality}` : municipality || loc.matched_name;
    const key = loc.psgc_code || `${name}:${loc.latitude}:${loc.longitude}`;
    if (!items.has(key)) items.set(key, { name, confidence: loc.confidence });
  }
  for (const ambiguous of row.ner_ambiguous_locations || []) {
    const candidates = ambiguous.candidates || [];
    if (!candidates.length) continue;
    const key = candidates.map((candidate) => candidate.psgc_code).sort().join(",");
    const names = [...new Set(candidates.map((candidate) => candidate.name))];
    const name = names.length === 1 ? names[0] : ambiguous.location_text;
    items.set(`ambiguous:${key}`, { name: `${name} (location unresolved)`, confidence: "Unresolved" });
  }
  return [...items.values()];
}

export async function loadSpatiotemporalData(posts, { signal, onProgress } = {}) {
  const postIds = [...new Set(posts.map((post) => post._id).filter(Boolean))];
  signal?.throwIfAborted();
  if (postIds.length === 0) return { rows: [], events: [], total: 0 };

  async function request(path) {
    signal?.throwIfAborted();
    const response = await fetch(`${API_BASE}/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ post_ids: postIds }),
      signal,
    });
    const data = await response.json();
    signal?.throwIfAborted();
    if (!response.ok) throw new Error(data.detail || "Failed to retrieve spatiotemporal information.");
    return data;
  }

  onProgress?.("Extracting location and time from the loaded posts...");
  let batch;
  do {
    batch = await request("process-all?limit=100");
    if (batch.processed === 0 && batch.remaining > 0) {
      throw new Error("Spatiotemporal processing stopped before all loaded posts were processed. Please refresh.");
    }
    onProgress?.(`Extracting spatiotemporal information: ${batch.remaining} posts remaining.`);
  } while (batch.remaining > 0);

  onProgress?.("Loading spatiotemporal information and event clusters...");
  return request("results");
}
