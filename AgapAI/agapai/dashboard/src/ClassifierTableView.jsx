import { useEffect, useMemo, useState } from "react";
import { matchesDisaster, matchesHandle, matchesRole } from "./filters";

const PAGE_SIZE = 100;

const fmtResources = (obj) => {
  const entries = Object.entries(obj || {});
  return entries.length
    ? entries.map(([k]) => k).join(", ")
    : "None";
};

const fmtLabel = (row, task) =>
  row[task] != null ? String(row[task]) : "None";

export default function ClassifierTableView({
  posts, results, loading, status, onClassify, onCancel, filters,
}) {
  const [page, setPage] = useState(1);
  const postById = useMemo(() => new Map(posts.map((p) => [p._id, p])), [posts]);
  const filteredResults = useMemo(() => results.filter((r) => (
    matchesDisaster(r.disaster, filters)
    && matchesRole(r.role, filters)
    && matchesHandle(postById.get(r._id)?.author_handle, filters.authorHandleQuery)
  )), [results, filters, postById]);
  const totalPages = Math.max(1, Math.ceil(filteredResults.length / PAGE_SIZE));
  const current = Math.min(page, totalPages);
  const pageRows = filteredResults.slice((current - 1) * PAGE_SIZE, current * PAGE_SIZE);

  useEffect(() => { setPage(1); }, [results, filters]);

  return (
    <>
      <section className="toolbar">
        <div className="load-controls">
          <button type="button" onClick={onClassify} disabled={loading || posts.length === 0}>
            {loading ? "Classifying..." : `Re-run Classifier on ${posts.length} loaded posts`}
          </button>
          <button type="button" className="cancel-button" onClick={onCancel} disabled={!loading}>
            Cancel
          </button>
        </div>
      </section>
      {status && <div className="status loading">{status}</div>}

      <section className="table-actions">
        <div className="row-count">{filteredResults.length} classified posts</div>
        <div className="pager">
          <button type="button" onClick={() => setPage(current - 1)} disabled={current === 1}>Prev</button>
          <span>Page {current} of {totalPages}</span>
          <button type="button" onClick={() => setPage(current + 1)} disabled={current === totalPages}>Next</button>
        </div>
      </section>

      <section className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Text</th>
              <th>Handle</th>
              <th>Role</th>
              <th>Intent</th>
              <th>Urgency</th>
              <th>Disaster Type</th>
              <th>Offers / Donate</th>
              <th>Requests / Needs</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 ? (
              <tr><td colSpan={8} className="empty">No classifier results yet. Click Load Data and posts are classified automatically.</td></tr>
            ) : pageRows.map((r) => {
              const post = postById.get(r._id) || {};
              return (
                <tr key={r._id}>
                  <td>{post.text}</td>
                  <td>{post.author_handle || "None"}</td>
                  <td>{fmtLabel(r, "role")}</td>
                  <td>{fmtLabel(r, "intent")}</td>
                  <td>{fmtLabel(r, "urgency")}</td>
                  <td>{fmtLabel(r, "disaster")}</td>
                  <td>{fmtResources(r.offers)}</td>
                  <td>{fmtResources(r.needs)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>
    </>
  );
}