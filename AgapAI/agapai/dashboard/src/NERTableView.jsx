import { useEffect, useMemo, useState } from "react";
import { filterEvents, postMatches } from "./filters";
import { locationDisplayItems } from "./spatiotemporal";

const PAGE_SIZE = 100;

function splitStrictDateTime(value) {
  if (!value) return { date: "None", time: "None" };

  const match = String(value).match(/^(\d{4}-\d{2}-\d{2})(?:[T\s](\d{2}:\d{2}:\d{2}))?/);
  if (!match) return { date: "None", time: "None" };

  return {
    date: match[1],
    time: match[2] || "00:00:00",
  };
}

function getStrictDateTime(row) {
  if (row.ner_event_date && row.ner_event_time) {
    return { date: row.ner_event_date, time: row.ner_event_time };
  }

  const extracted = row.ner_datetime?.expressions?.[0]?.normalized_datetime;
  return splitStrictDateTime(row.ner_event_datetime || extracted || row.created_at);
}

function getEventDateTime(event) {
  if (event.event_date && event.event_time) {
    return { date: event.event_date, time: event.event_time };
  }

  return splitStrictDateTime(event.event_datetime);
}

function formatFinalLocations(row) {
  return locationDisplayItems(row).map((item) => item.name).join("; ") || "None";
}

function formatConfidences(row) {
  return locationDisplayItems(row).map((item) => item.confidence).join("; ") || "None";
}

const MAX_CANDIDATES_SHOWN = 5;

function describeCandidate(candidate) {
  if (candidate.parent_city_municipality) return candidate.parent_city_municipality;
  return `${candidate.name} (${candidate.type})`;
}

function formatAmbiguous(ambiguousLocations) {
  return (ambiguousLocations || [])
    .filter((amb) => amb.candidates?.length > 0)
    .map((amb) => {
      const shown = amb.candidates.slice(0, MAX_CANDIDATES_SHOWN).map(describeCandidate);
      const suffix = amb.candidates.length > MAX_CANDIDATES_SHOWN ? ", ..." : "";
      return `${amb.location_text} (${amb.candidate_count} matches: ${shown.join(", ")}${suffix})`;
    })
    .join("; ") || "None";
}

function getEventAuthors(event) {
  const authors = [...new Set(
    (event.posts || [])
      .map((post) => post.author_handle || post.posted_by)
      .filter(Boolean)
  )];

  return authors.length ? authors.join(", ") : "Unknown";
}

export default function NERTableContent({
  rows,
  events,
  loading,
  onRefresh,
  hasLoadedPosts,
  highlightedPostIds,
  subView,
  setSubView,
  selectedEventId,
  setSelectedEventId,
  filters,
}) {
  const [page, setPage] = useState(1);

  const filteredEvents = useMemo(() => filterEvents(events, filters), [events, filters]);
  const filteredRows = useMemo(() => rows.filter((row) => postMatches(row, filters)), [rows, filters]);
  const totalPages = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageRows = filteredRows.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const visibleEvents = useMemo(() => {
    if (!selectedEventId) return filteredEvents;
    return filteredEvents.filter((event) => event.event_id === selectedEventId);
  }, [filteredEvents, selectedEventId]);

  useEffect(() => {
    setPage(1);
  }, [rows, filters]);

  useEffect(() => {
    if (subView !== "clusters" || !selectedEventId) return;

    window.requestAnimationFrame(() => {
      document.getElementById(`cluster-${selectedEventId}`)?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    });
  }, [subView, selectedEventId, events]);

  return (
    <>
      <section className="toolbar">
        <div className="view-buttons">
          <button
            type="button"
            className={subView === "posts" ? "active" : ""}
            onClick={() => {
              setSelectedEventId(null);
              setSubView("posts");
            }}
          >
            All Posts
          </button>
          <button
            type="button"
            className={subView === "clusters" ? "active" : ""}
            onClick={() => {
              setSelectedEventId(null);
              setSubView("clusters");
            }}
          >
            Clustered Events
          </button>
        </div>
        <button type="button" onClick={onRefresh} disabled={loading || !hasLoadedPosts}>
          Refresh Spatiotemporal Info
        </button>
      </section>

      {subView === "posts" && (
        <section className="table-actions" aria-label="Spatiotemporal table controls">
          <div className="row-count">{filteredRows.length} loaded posts</div>
          <div className="pager">
            <button type="button" onClick={() => setPage(currentPage - 1)} disabled={currentPage === 1 || loading}>Prev</button>
            <span>Page {currentPage} of {totalPages}</span>
            <button type="button" onClick={() => setPage(currentPage + 1)} disabled={currentPage === totalPages || loading}>Next</button>
          </div>
        </section>
      )}

      {subView === "posts" && <section className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Text</th>
              <th>Handle</th>
              <th>Posted By</th>
              <th>Final Location(s)</th>
              <th>Confidence</th>
              <th>Ambiguous Locations</th>
              <th>Date</th>
              <th>Time (PHT)</th>
            </tr>
          </thead>
          <tbody>
            {filteredRows.length === 0 && !loading ? (
              <tr><td colSpan={8} className="empty">No spatiotemporal results for the loaded posts. Select dates and click Load Data to begin.</td></tr>
            ) : (
              pageRows.map((row) => (
                <tr key={row._id}>
                  <td>{row.text}</td>
                  <td>{row.author_handle || "None"}</td>
                  <td>{row.posted_by || "None"}</td>
                  <td>{formatFinalLocations(row)}</td>
                  <td>{formatConfidences(row)}</td>
                  <td>{formatAmbiguous(row.ner_ambiguous_locations)}</td>
                  <td>{getStrictDateTime(row).date}</td>
                  <td>{getStrictDateTime(row).time}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </section>}

      {subView === "clusters" && (
        <section className="event-list">
          {selectedEventId && (
            <div className="cluster-filter">
              Showing cluster {selectedEventId}
              <button type="button" onClick={() => setSelectedEventId(null)}>
                Show All Events
              </button>
            </div>
          )}

          {visibleEvents.length === 0 ? (
            <div className="empty">No clustered events found.</div>
          ) : (
            visibleEvents.map((event) => {
              const dateTime = getEventDateTime(event);
              const eventLocations = [...new Set((event.locations || []).map((location) => location.place).filter(Boolean))];

              return (
                <article
                  key={event.event_id}
                  id={`cluster-${event.event_id}`}
                  className={`event-card ${event.event_id === selectedEventId ? "selected-event" : ""}`}
                >
                  <div className="event-card-header">
                    <div>
                      <h3>{event.disaster_type || "Unknown Disaster"}</h3>
                      <p>{eventLocations.length > 1
                        ? `Locations: ${eventLocations.join("; ")}`
                        : eventLocations[0] || event.place || "Unknown location"}</p>
                    </div>
                    <span className="event-pill">{event.post_count} post{event.post_count === 1 ? "" : "s"}</span>
                  </div>

                  <div className="event-meta">
                    <span>Event ID: {event.event_id}</span>
                    <span>Date: {dateTime.date}</span>
                    <span>Time (PHT): {dateTime.time}</span>
                    <span>Authors: {getEventAuthors(event)}</span>
                  </div>

                  <div className="cluster-posts">
                    {(event.posts || []).map((post) => (
                      <div
                        key={post._id}
                        className={highlightedPostIds?.includes(post._id) ? "cluster-post highlighted-cluster-post" : "cluster-post"}
                      >
                        <strong>{post.author_handle || post.posted_by || "Unknown"}</strong>
                        <p>{post.text || "No text available."}</p>
                      </div>
                    ))}
                  </div>
                </article>
              );
            })
          )}
        </section>
      )}
    </>
  );
}
