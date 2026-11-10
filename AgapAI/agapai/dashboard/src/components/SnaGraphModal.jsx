import { useEffect, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

const API_BASE = "http://127.0.0.1:8000";
const BRIDGE_COLOR = "#fbbf24";
const TREE_COLOR = "#475569";

function formatNeeds(victim, target) {
  const needs = target?.needed_items ?? victim?.needed_items ?? victim?.needs;
  const items = Array.isArray(needs)
    ? needs.filter(Boolean)
    : String(needs || "").trim() ? [String(needs).trim()] : [];
  const isUnspecified =
    items.length === 0 || items.some((item) => String(item).trim().toLowerCase() === "unspecified");
  if (isUnspecified) return "Unspecified (No volunteers fetched)";
  return items.join(", ");
}

function formatList(items) {
  if (Array.isArray(items)) {
    const text = items.filter(Boolean).join(", ");
    return text || "None specified.";
  }
  const text = String(items || "").trim();
  return text || "None specified.";
}

function resolveEndpointId(endpoint) {
  return typeof endpoint === "object" && endpoint !== null ? endpoint.id : endpoint;
}

export default function SnaGraphModal({ victim, onClose }) {
  const graphRef = useRef(null);
  const [graph, setGraph] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedNode, setSelectedNode] = useState(null);

  const victimDid = victim?.author_did || victim?.did;

  useEffect(() => {
    setGraph(null);
    setSelectedNode(null);
    setError("");

    if (!victimDid) {
      setError("This incident has no author DID.");
      return;
    }

    let cancelled = false;
    setLoading(true);
    fetch(`${API_BASE}/api/sna/graph/${encodeURIComponent(victimDid)}${victim?._id ? `?post_id=${encodeURIComponent(victim._id)}` : ""}`)
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not build the SNA graph.");
        if (!cancelled) setGraph(data);
      })
      .catch((fetchError) => {
        if (!cancelled) setError(fetchError.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [victimDid, victim?._id]);

  useEffect(() => {
    graphRef.current?.d3Force("charge")?.strength(-180);
  }, [graph]);

  if (!victim) return null;

  function selectNode(node) {
    setSelectedNode(node);
  }

  const nodes = graph?.nodes || [];
  const links = graph?.links || [];
  const nodesById = new Map(nodes.map((node) => [node.id, node]));

  function isBridgeLink(link) {
    if (typeof link.isBridge === "boolean") return link.isBridge;
    const sourceNode = nodesById.get(resolveEndpointId(link.source));
    const targetNode = nodesById.get(resolveEndpointId(link.target));
    return Boolean(
      sourceNode && targetNode && sourceNode.role !== "Victim" && sourceNode.role !== "Volunteer"
        && targetNode.role !== "Victim" && targetNode.role !== "Volunteer"
    );
  }

  return (
    <div className="sna-modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="sna-modal" role="dialog" aria-modal="true" aria-labelledby="sna-graph-title">
        <header className="sna-modal-header">
          <div>
            <p className="eyebrow">Social network analysis </p>
            <h2 id="sna-graph-title">{victim.author_handle || victim.author_did || victim.did}</h2>
            <p>
              {victim.urgency_level || victim.urgency || "Low"} priority · needs: {formatNeeds(victim, graph?.target)}
            </p>
          </div>
          <button type="button" className="icon-button" aria-label="Close graph" onClick={onClose}>×</button>
        </header>
        <div className="sna-legend" aria-label="Graph legend">
          <span><i className="legend-dot" style={{ background: "#ff2a2a" }} /> Victim</span>
          <span><i className="legend-dot" style={{ background: "#00b85c" }} /> Volunteer</span>
          <span><i className="legend-dot" style={{ background: "#0099cc" }} /> Mutual Connection</span>
          <span><i className="legend-dot" style={{ background: "#81d873" }} /> 1-Way Link</span>
        </div>
        {loading && <div className="sna-status" role="status">Building the social graph...</div>}
        {error && <div className="sna-status" role="alert">{error}</div>}
        {!loading && !error && graph?.message && <div className="sna-status" role="status">{graph.message}</div>}
        {!loading && !error && graph && (
          <div className="sna-modal-body">
            <div className="sna-graph-canvas">
              <ForceGraph2D
                ref={graphRef}
                graphData={{ nodes, links }}
                dagMode={null}
                d3VelocityDecay={0.35}
                d3AlphaDecay={0.04}
                width={Math.max(560, Math.min(760, window.innerWidth - 420))}
                height={Math.min(650, Math.max(430, window.innerHeight - 260))}
                nodeRelSize={6}
                nodeVal={(node) => node.val || 3}
                nodeColor={(node) => node.color}
                nodeLabel={(node) => `@${node.handle}`}
                nodeCanvasObject={(node, ctx) => {
                  const baseRadius = Math.sqrt(node.val || 3) * 2.2;
                  const radius = node.id === selectedNode?.id ? baseRadius + 2 : baseRadius;
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
                  ctx.fillStyle = node.color;
                  ctx.fill();
                  if (node.id === selectedNode?.id) {
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = "#0f172a";
                    ctx.stroke();
                  }
                }}
                nodePointerAreaPaint={(node, color, ctx) => {
                  const baseRadius = Math.sqrt(node.val || 3) * 2.2;
                  ctx.fillStyle = color;
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, baseRadius + 3, 0, 2 * Math.PI, false);
                  ctx.fill();
                }}
                linkWidth={(link) => (isBridgeLink(link) ? 3.5 : 1)}
                linkColor={(link) => (isBridgeLink(link) ? BRIDGE_COLOR : TREE_COLOR)}
                linkLabel={(link) => link.label || `Score: ${link.score ?? 0}`}
                linkDirectionalArrowLength={4}
                linkDirectionalArrowRelPos={1}
                onNodeClick={selectNode}
                cooldownTicks={100}
              />
            </div>
            <aside className="sna-inspector" aria-label="Node inspector">
              {!selectedNode ? (
                <p className="sna-inspector-empty">Click a node to inspect its role and connection trust score.</p>
              ) : (
                <>
                  <p className="eyebrow">{selectedNode.role}</p>
                  <h3>@{selectedNode.handle || selectedNode.did}</h3>
                  {selectedNode.role === "Volunteer" && (
                    <>
                      <p className="sna-trust-score">{selectedNode.closeness_score ?? 0} <span>/ 100</span></p>
                      <p className="sna-score-badge">{selectedNode.score_badge || "No score available"}</p>
                      <p className="sna-score-breakdown">{selectedNode.score_breakdown || "No path explanation available for this node."}</p>
                      <div className="sna-inspector-section">
                        <p className="eyebrow">Original Post</p>
                        <p className="sna-post-text">{selectedNode.post_text || "No post text available."}</p>
                      </div>
                      <div className="sna-inspector-section">
                        <p className="eyebrow">Items Offered</p>
                        <p className="sna-post-text">{formatList(selectedNode.items)}</p>
                      </div>
                    </>
                  )}
                  {selectedNode.role !== "Volunteer" && selectedNode.role !== "Victim" && (
                    <p className="sna-score-breakdown">
                      Bridges the victim's network toward matched volunteers as a{" "}
                      {selectedNode.role === "Mutual Connection" ? "mutual (two-way) connection" : "one-way follower link"}.
                    </p>
                  )}
                  {selectedNode.role !== "Volunteer" && selectedNode.role !== "Victim" && (
                    <div className="sna-inspector-section">
                      <p className="eyebrow">Post</p>
                      <p className="sna-post-text">{selectedNode.post_text || "No post text available."}</p>
                    </div>
                  )}
                  {selectedNode.role === "Victim" && (
                    <>
                      <p className="sna-score-breakdown">{selectedNode.score_breakdown}</p>
                      <div className="sna-inspector-section">
                        <p className="eyebrow">Original Post</p>
                        <p className="sna-post-text">{selectedNode.post_text || "No post text available."}</p>
                      </div>
                      <div className="sna-inspector-section">
                        <p className="eyebrow">Specific Needs</p>
                        <p className="sna-post-text">{formatList(selectedNode.items)}</p>
                      </div>
                    </>
                  )}
                </>
              )}
            </aside>
          </div>
        )}
        <p className="sna-modal-hint">Click any node for its role and trust score.</p>
      </section>
    </div>
  );
}
