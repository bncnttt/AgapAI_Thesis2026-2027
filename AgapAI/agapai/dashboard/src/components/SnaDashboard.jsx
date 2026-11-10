import { useEffect, useState } from "react";
import SnaGraphModal from "./SnaGraphModal";
import "./SnaDashboard.css";

const URGENCY_ORDER = { High: 0, Medium: 1, Low: 2 };
const URGENCY_STYLE = {
  High: { color: "#dc2626", icon: "🔴", label: "High" },
  Medium: { color: "#ca8a04", icon: "🟡", label: "Med" },
  Low: { color: "#16a34a", icon: "🟢", label: "Low" },
};

function items(value) {
  if (Array.isArray(value)) return value.join(", ");
  return String(value || "requested supplies");
}

export default function SnaDashboard({ posts = [] }) {
  const [incidents, setIncidents] = useState([]);
  const [selectedVictim, setSelectedVictim] = useState(null);

  useEffect(() => {
    const victims = posts
      .filter((post) => String(post.role || "").toLowerCase() === "victim" && post.help_seeking === true)
      .sort((first, second) => (
        (URGENCY_ORDER[first.urgency_level || first.urgency] ?? 2)
        - (URGENCY_ORDER[second.urgency_level || second.urgency] ?? 2)
      ));
    setIncidents(victims);
    setSelectedVictim(null);
  }, [posts]);

  return (
    <main className="sna-dashboard">
      <header className="sna-header"><div><p className="eyebrow">AgapAI / response coordination</p><h1>SNA Dispatch Console</h1><p>Item-matched volunteers, scored by social-bridge proximity to each victim.</p></div><span className="sna-sync-state">{`${incidents.length} victim incidents synced`}</span></header>
      <div className="sna-layout">
        <section className="sna-panel queue-panel full-width">
          <div className="panel-heading"><div><p className="eyebrow">Live queue</p><h2>Victim incidents</h2></div><strong>{incidents.length}</strong></div>
          {incidents.length === 0 ? (
            <p className="sna-empty">No classified victim incidents found.</p>
          ) : (
            <div className="sna-table-wrap">
              <table className="sna-table">
                <thead>
                  <tr>
                    <th>Victim</th>
                    <th>Urgency Level</th>
                    <th>Needs</th>
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
                        onClick={() => setSelectedVictim(incident)}
                      >
                        <td>
                          <b>{incident.author_handle || incident.author_did || "Unknown victim"}</b>
                          <small>{incident.location_name || incident.location || "Location unknown"}</small>
                        </td>
                        <td><span className="urgency-badge" style={{ color: style.color }}>{style.icon} {style.label}</span></td>
                        <td>{items(incident.needed_items || incident.needs)}</td>
                        <td>
                          <button
                            type="button"
                            className="inspect-button"
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
        <SnaGraphModal victim={selectedVictim} onClose={() => setSelectedVictim(null)} />
      )}
    </main>
  );
}
