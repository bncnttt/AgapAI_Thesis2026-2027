const API_BASE = "http://127.0.0.1:8000/classifier";
const BATCH = 100;

export async function classifyPosts(posts, { signal, onProgress } = {}) {
  const items = posts
    .filter((p) => p._id && p.text)
    .map((p) => ({ id: p._id, text: p.text }));
  console.log(`[classifier] ${items.length} of ${posts.length} posts have _id + text and will be sent`);
  const rows = [];
  for (let i = 0; i < items.length; i += BATCH) {
    signal?.throwIfAborted();
    onProgress?.(`Classifying ${Math.min(i + BATCH, items.length)} / ${items.length}...`);
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items: items.slice(i, i + BATCH) }),
      signal,
    });
    const data = await res.json();
    console.log(`[classifier] batch ${i / BATCH + 1}: HTTP ${res.status}`, data);
    signal?.throwIfAborted();
    if (!res.ok) throw new Error(data.detail || "Classification failed.");
    rows.push(...data.rows);
  }
  console.log(`[classifier] finished: ${rows.length} rows`, rows);
  return rows;
}