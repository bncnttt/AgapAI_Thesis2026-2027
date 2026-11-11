import { useCallback, useEffect, useState } from "react";
import SnaGraphModal from "./SnaGraphModal";
import "./SnaDashboard.css";

const URGENCY_ORDER = { "Very High": 0, High: 1, Medium: 2, Low: 3 };
const URGENCY_STYLE = {
  "Very High": { color: "#991b1b", icon: "●", label: "Very High" },
  High: { color: "#dc2626", icon: "🔴", label: "High" },
  Medium: { color: "#ca8a04", icon: "🟡", label: "Med" },
  Low: { color: "#16a34a", icon: "🟢", label: "Low" },
};

function items(value) {
  if (!value) return "None specified";
  if (Array.isArray(value)) return value.filter(Boolean).join(", ") || "None specified";
  if (typeof value === "object") return Object.keys(value).filter(Boolean).join(", ") || "None specified";
  return String(value);
}

const API_BASE = "http://127.0.0.1:8000";
const MATCH_STORAGE_KEY = "agapai.sna.matches.v1";
// Graphs survive SNA component remounts; only match summaries go into storage.
const graphCache = {};
const pendingMatches = new Map();

function calculateMatch(incident) {
  const key = incident._id;
  if (graphCache[key]) return Promise.resolve(graphCache[key]);
  if (!pendingMatches.has(key)) {
    const request = fetch(`${API_BASE}/api/sna/graph/${encodeURIComponent(incident.author_did)}?post_id=${encodeURIComponent(key)}`)
      .then(async (response) => {
        const graph = await response.json();
        if (!response.ok) throw new Error(graph.detail || "Could not calculate the volunteer match.");
        return saveMatch(key, graph);
      })
      .finally(() => pendingMatches.delete(key));
    pendingMatches.set(key, request);
  }
  return pendingMatches.get(key);
}

function restoreMatches() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(MATCH_STORAGE_KEY) || "{}");
    return saved && typeof saved === "object" && !Array.isArray(saved)
      ? { ...saved, ...graphCache } : { ...graphCache };
  } catch {
    return { ...graphCache };
  }
}

function saveMatch(postId, graph) {
  const entry = { graph, match: graph.jaccard?.best_match || null };
  graphCache[postId] = entry;
  try {
    const summaries = restoreMatches();
    const saved = Object.fromEntries(Object.entries(summaries).map(([id, result]) => (
      [id, { match: result.match || null }]
    )));
    sessionStorage.setItem(MATCH_STORAGE_KEY, JSON.stringify(saved));
  } catch {
    // In-memory results still survive tab navigation if storage is unavailable.
  }
  return entry;
}

export default function SnaDashboard({ posts, startDate = "", endDate = "", searchTerm }) {
  const [incidents, setIncidents] = useState([]);
  const [selectedVictim, setSelectedVictim] = useState(null);
  const [error, setError] = useState("");
  const [fallbackSearch, setFallbackSearch] = useState("");
  const [matchResults, setMatchResults] = useState(restoreMatches);
  const rememberGraph = useCallback((postId, graph) => {
    const entry = saveMatch(postId, graph);
    setMatchResults((previous) => ({ ...previous, [postId]: entry }));
  }, []);

  useEffect(() => {
    let cancelled = false;
    // A single queue avoids simultaneous live graph crawls. Results appear
    // progressively and completed/in-flight work is reused on tab navigation.
    async function calculateQueue() {
      const saved = restoreMatches();
      for (const incident of incidents) {
        if (cancelled) break;
        const key = incident._id;
        if (saved[key]) {
          setMatchResults((previous) => ({ ...previous, [key]: saved[key] }));
          continue;
        }
        setMatchResults((previous) => ({ ...previous, [key]: { loading: true } }));
        try {
          const entry = await calculateMatch(incident);
          if (!cancelled) setMatchResults((previous) => ({ ...previous, [key]: entry }));
        } catch (requestError) {
          if (!cancelled) setMatchResults((previous) => ({
            ...previous, [key]: { error: requestError.message },
          }));
        }
      }
    }
    calculateQueue();
    return () => { cancelled = true; };
  }, [incidents]);

  useEffect(() => {
    if (searchTerm !== undefined) return undefined;
    const input = document.querySelector('input[placeholder="Search by author handle"]');
    if (!input) return undefined;
    const syncSearch = () => setFallbackSearch(input.value || "");
    syncSearch();
    input.addEventListener("input", syncSearch);
    return () => input.removeEventListener("input", syncSearch);
  }, [searchTerm]);

  useEffect(() => {
    const params = new URLSearchParams();
    // Never send an invalid date interval to the SNA API. The Posts view can
    // still contain the last successfully loaded incident set while a user is
    // editing the date controls, so SNA falls back to its available queue.
    if (startDate && endDate && startDate <= endDate) {
      params.set("start", startDate);
      params.set("end", endDate);
    }
    const activeSearch = String(searchTerm ?? fallbackSearch).trim();
    if (activeSearch) params.set("author", activeSearch);
    const query = params.toString();
    const incidentsUrl = `${API_BASE}/api/sna/incidents${query ? `?${query}` : ""}`;
    let cancelled = false;
    setError("");
    fetch(incidentsUrl)
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) {
          console.error("SNA incidents request failed", { url: incidentsUrl, status: response.status, data });
          throw new Error("Failed to load SNA victim incidents.");
        }
        const loadedIds = Array.isArray(posts) ? new Set(posts.map((post) => String(post._id))) : null;
        if (!cancelled) setIncidents((data.incidents || [])
          .filter((incident) => !loadedIds || loadedIds.has(String(incident._id)))
          .sort((first, second) => (
            (URGENCY_ORDER[first.urgency_level || first.urgency] ?? 3)
            - (URGENCY_ORDER[second.urgency_level || second.urgency] ?? 3)
          )));
      })
      .catch((requestError) => {
        if (!cancelled) {
          console.error("SNA incidents fetch error", requestError);
          setError(`Could not reach SNA API at ${incidentsUrl}. ${requestError.message || ""}`.trim());
        }
      });
    setSelectedVictim(null);
    return () => { cancelled = true; };
  }, [posts, startDate, endDate, searchTerm, fallbackSearch]);

  return (
    <main className="sna-dashboard">
      <header className="sna-header"><div><p className="eyebrow">AgapAI / response coordination</p><h1>SNA Dispatch Console</h1><p>Item-matched volunteers, scored by social-bridge proximity to each victim.</p></div><span className="sna-sync-state">{`${incidents.length} victim incidents synced`}</span></header>
      <div className="sna-layout">
        <section className="sna-panel queue-panel full-width">
          <div className="panel-heading"><div><p className="eyebrow">Live queue</p><h2>Victim incidents</h2></div><strong>{incidents.length}</strong></div>
          {incidents.length === 0 ? (
            <p className="sna-empty">{error || "No classified victim incidents found."}</p>
          ) : (
            <div className="sna-table-wrap">
              <table className="sna-table">
                <thead>
                  <tr>
                    <th>Victim</th>
                    <th>Urgency Level</th>
                    <th>Needs</th>
                    <th>Matched volunteer</th>
                    <th aria-label="Actions" />
                  </tr>
                </thead>
                <tbody>
                  {incidents.map((incident) => {
                    const urgency = incident.urgency_level || incident.urgency || "Low";
                    const style = URGENCY_STYLE[urgency] || URGENCY_STYLE.Low;
                    const key = incident._id || incident.author_did;
                    return (
                      <tr
                        key={key}
                        className={selectedVictim?._id === incident._id ? "selected" : ""}
                        onClick={() => {
                          if (matchResults[key] && !matchResults[key].loading) setSelectedVictim(incident);
                        }}
                      >
                        <td>
                          <b>{incident.author_handle || incident.author_did || "Unknown victim"}</b>
                          <small>{incident.location_name || incident.location || "Location unknown"}</small>
                        </td>
                        <td><span className="urgency-badge" style={{ color: style.color }}>{style.icon} {style.label}</span></td>
                        <td>{items(incident.needed_items || incident.needs)}</td>
                        <td onClick={(event) => event.stopPropagation()}>
                          {matchResults[key]?.error ? (
                            <span role="alert" title={matchResults[key].error}>Match calculation failed</span>
                          ) : matchResults[key]?.loading ? "Calculating..." : matchResults[key] ? (
                            matchResults[key].match ? (
                              <b className="sna-matched-volunteer">{matchResults[key].match.handle || matchResults[key].match.did}</b>
                            ) : "No eligible volunteer found"
                          ) : "Waiting to calculate..."}
                        </td>
                        <td>
                          <button
                            type="button"
                            className="inspect-button"
                            disabled={!matchResults[key] || matchResults[key].loading}
                            onClick={(event) => { event.stopPropagation(); setSelectedVictim(incident); }}
                          >
                            View Social Graph
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
      {selectedVictim && (
        <SnaGraphModal victim={selectedVictim} initialGraph={matchResults[selectedVictim._id]?.graph} onGraphReady={rememberGraph} onClose={() => setSelectedVictim(null)} />
      )}
    </main>
  );
}
