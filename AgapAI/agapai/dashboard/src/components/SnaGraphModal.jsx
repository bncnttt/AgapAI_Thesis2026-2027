import { useEffect, useMemo, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

function formatNeeds(victim, target) {
  const value = target?.needed_items ?? victim?.needed_items ?? victim?.needs;
  if (Array.isArray(value)) return value.filter(Boolean).join(", ") || "Unspecified";
  if (value && typeof value === "object") return Object.keys(value).filter(Boolean).join(", ") || "Unspecified";
  return String(value || "").trim() || "Unspecified";
}

function formatList(value) {
  if (Array.isArray(value)) return value.filter(Boolean).join(", ") || "None specified.";
  if (value && typeof value === "object") return Object.keys(value).filter(Boolean).join(", ") || "None specified.";
  return String(value || "").trim() || "None specified.";
}

function endpointId(value) {
  return typeof value === "object" && value !== null ? value.id : value;
}

function nodeColor(node) {
  if (node?.role === "Victim") return "#ef4444";
  if (node?.role === "Volunteer") return "#10b981";
  if (node?.role === "Mutual Connection") return "#0ea5e9";
  return "#86d97a";
}

function shortLabel(node) {
  const text = String(node?.handle || node?.did || node?.id || "");
  return text.length > 19 ? `${text.slice(0, 16)}...` : text;
}

function unique(values) {
  return [...new Set((values || []).filter(Boolean))];
}

function buildReadableGraph(graph) {
  if (!graph) return null;

  const width = 1180;
  const height = 650;

  const victimId = graph.target?.did;
  const best = graph.jaccard?.best_match;
  const volunteerId = best?.did;
  const victimNet = victimId ? graph.neighborhoods?.[victimId] : null;
  const volunteerNet = volunteerId ? graph.neighborhoods?.[volunteerId] : null;
  const path = best?.path || (victimId ? [victimId] : []);

  const nodeMap = new Map();
  const addNode = (node) => {
    if (node?.id && !nodeMap.has(node.id)) nodeMap.set(node.id, { ...node });
  };

  for (const node of graph.nodes || []) addNode(node);
  for (const view of Object.values(graph.hop_views || {})) {
    for (const node of view?.nodes || []) addNode(node);
  }
  for (const net of Object.values(graph.neighborhoods || {})) {
    for (const node of net?.nodes || []) addNode(node);
  }

  const fallback = (id, role = "1-Way Link") => ({
    id,
    did: id,
    handle: id,
    role,
    post_text: "No post text available.",
  });

  if (victimId && !nodeMap.has(victimId)) nodeMap.set(victimId, fallback(victimId, "Victim"));
  if (volunteerId && !nodeMap.has(volunteerId)) nodeMap.set(volunteerId, fallback(volunteerId, "Volunteer"));

  const victimFollowing = unique(victimNet?.following).slice(0, 5);
  const victimFollowers = unique(victimNet?.followers).slice(0, 5);
  const volunteerFollowing = unique(volunteerNet?.following).slice(0, 5);
  const volunteerFollowers = unique(volunteerNet?.followers).slice(0, 5);

  const leftSet = new Set([victimId, ...victimFollowing, ...victimFollowers].filter(Boolean));
  const rightSet = new Set([volunteerId, ...volunteerFollowing, ...volunteerFollowers].filter(Boolean));

  const visible = new Set([
    ...leftSet,
    ...rightSet,
    ...path,
  ]);

  for (const id of visible) {
    if (!nodeMap.has(id)) nodeMap.set(id, fallback(id));
  }

  const positions = new Map();
  const victimCenter = { x: 280, y: 325 };
  const volunteerCenter = { x: 900, y: 325 };

  if (victimId) positions.set(victimId, victimCenter);
  if (volunteerId) positions.set(volunteerId, volunteerCenter);

  // Five clear top/bottom slots. Placement communicates relationship,
  // so local edges do not need arrowheads.
  const slots = [-170, -85, 0, 85, 170];

  victimFollowing.forEach((id, i) => {
    positions.set(id, { x: victimCenter.x + slots[i], y: 120 });
  });
  victimFollowers.forEach((id, i) => {
    if (!positions.has(id)) positions.set(id, { x: victimCenter.x + slots[i], y: 530 });
  });

  volunteerFollowing.forEach((id, i) => {
    positions.set(id, { x: volunteerCenter.x + slots[i], y: 120 });
  });
  volunteerFollowers.forEach((id, i) => {
    if (!positions.has(id)) positions.set(id, { x: volunteerCenter.x + slots[i], y: 530 });
  });

  // Only interior selected-path nodes occupy the clean middle corridor.
  const innerPath = path.slice(1, -1);
  innerPath.forEach((id, i) => {
    const step = (volunteerCenter.x - victimCenter.x) / (innerPath.length + 1);
    positions.set(id, {
      x: victimCenter.x + step * (i + 1),
      y: victimCenter.y,
    });
  });

  // Any node shared by follower/following lists gets a deterministic side position.
  let fallbackIndex = 0;
  for (const id of visible) {
    if (!positions.has(id)) {
      positions.set(id, { x: 590, y: 195 + fallbackIndex * 55 });
      fallbackIndex += 1;
    }
  }

  const nodes = [...visible].map((id) => ({
    ...nodeMap.get(id),
    id,
    x: positions.get(id).x,
    y: positions.get(id).y,
  }));

  const edges = [];
  const seen = new Set();

  function addEdge(source, target, kind, label, highlighted = false) {
    if (!source || !target || source === target || !visible.has(source) || !visible.has(target)) return;
    const key = `${source}|${target}|${kind}`;
    if (seen.has(key)) return;
    seen.add(key);
    edges.push({ source, target, kind, label, highlighted });
  }

  // Local fan edges.
  victimFollowing.forEach((id) => addEdge(victimId, id, "local", "Victim follows"));
  victimFollowers.forEach((id) => addEdge(id, victimId, "local", "Follows Victim"));
  volunteerFollowing.forEach((id) => addEdge(volunteerId, id, "local", "Volunteer follows"));
  volunteerFollowers.forEach((id) => addEdge(id, volunteerId, "local", "Follows Volunteer"));

  // IMPORTANT: only show cross-network bridge edges from left neighborhood
  // to right neighborhood. Do not draw left-left/right-right extra edges.
  for (const raw of graph.neighborhood_cross_links || []) {
    const source = endpointId(raw.source);
    const target = endpointId(raw.target);
    const crossesSides =
      (leftSet.has(source) && rightSet.has(target))
      || (rightSet.has(source) && leftSet.has(target));

    if (crossesSides) {
      addEdge(source, target, "cross", raw.label || "Cross-network connection");
    }
  }

  // Selected path is always the strongest layer.
  for (let i = 0; i < path.length - 1; i += 1) {
    addEdge(path[i], path[i + 1], "path", "Selected path", true);
  }

  return {
    width,
    height,
    nodes,
    edges,
    victimId,
    volunteerId,
    best,
    victimNet,
    volunteerNet,
  };
}

function edgeCurve(source, target, kind) {
  const sx = source.x;
  const sy = source.y;
  const tx = target.x;
  const ty = target.y;

  if (kind === "local") {
    // Smooth fan curve toward the anchor.
    const cx = (sx + tx) / 2;
    const cy = (sy + ty) / 2 + (sy < ty ? -28 : 28);
    return `M ${sx} ${sy} Q ${cx} ${cy} ${tx} ${ty}`;
  }

  if (kind === "cross") {
    // Cross-network connections curve around the middle rather than
    // cutting straight through labels/nodes.
    const midX = (sx + tx) / 2;
    const bend = sy <= 325 && ty <= 325 ? -80 : sy >= 325 && ty >= 325 ? 80 : 0;
    return `M ${sx} ${sy} C ${midX - 80} ${sy + bend}, ${midX + 80} ${ty + bend}, ${tx} ${ty}`;
  }

  return `M ${sx} ${sy} L ${tx} ${ty}`;
}

export default function SnaGraphModal({ victim, onClose, initialGraph, onGraphReady }) {
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

    if (initialGraph) {
      setGraph(initialGraph);
      setSelectedNode(initialGraph.target || null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    const suffix = victim?._id ? `?post_id=${encodeURIComponent(victim._id)}` : "";

    fetch(`${API_BASE}/api/sna/graph/${encodeURIComponent(victimDid)}${suffix}`)
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Could not build the SNA graph.");
        if (!cancelled) {
          setGraph(data);
          setSelectedNode(data?.target || null);
          onGraphReady?.(victim._id, data);
        }
      })
      .catch((fetchError) => !cancelled && setError(fetchError.message))
      .finally(() => !cancelled && setLoading(false));

    return () => { cancelled = true; };
  }, [victimDid, victim?._id, initialGraph, onGraphReady]);

  const layout = useMemo(() => buildReadableGraph(graph), [graph]);
  const nodesById = useMemo(
    () => new Map((layout?.nodes || []).map((node) => [node.id, node])),
    [layout]
  );

  if (!victim) return null;

  return (
    <div className="sna-modal-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section className="sna-modal sna-readable-modal" role="dialog" aria-modal="true">
        <header className="sna-modal-header">
          <div>
            <p className="eyebrow">Social network analysis</p>
            <h2>{victim.author_handle || victim.author_did || victim.did}</h2>
            <p>{victim.urgency_level || victim.urgency || "Low"} priority · needs: {formatNeeds(victim, graph?.target)}</p>
          </div>
          <button type="button" className="icon-button" onClick={onClose}>×</button>
        </header>

        <div className="sna-graph-summary">
          <div>
            <b>Victim network</b>
            <span>{layout?.victimNet ? `${layout.victimNet.following_shown} · ${layout.victimNet.followers_shown} followers` : "No network found"}</span>
          </div>
          <div>
            <b>Volunteer network</b>
            <span>{layout?.volunteerNet ? `${layout.volunteerNet.following_shown} · ${layout.volunteerNet.followers_shown} followers` : "No network found"}</span>
          </div>
        </div>

        <div className="sna-legend">
          <span><i className="legend-dot" style={{ background: "#ef4444" }} /> Victim</span>
          <span><i className="legend-dot" style={{ background: "#10b981" }} /> Volunteer</span>
          <span><i className="legend-dot" style={{ background: "#0ea5e9" }} /> Mutual</span>
          <span><i className="legend-dot" style={{ background: "#86d97a" }} /> 1-way</span>
          <span><i className="legend-line bridge-line" /> Cross-network bridge</span>
          <span><i className="legend-line selected-line" /> Best path</span>
        </div>

        {loading && <div className="sna-status">Building the social graph...</div>}
        {error && <div className="sna-status sna-status-error">{error}</div>}

        {!loading && !error && layout && (
          <div className="sna-modal-body readable-body">
            <div className="sna-readable-graph">
              <svg viewBox={`0 0 ${layout.width} ${layout.height}`} className="sna-readable-svg">
                <text x="280" y="36" textAnchor="middle" className="graph-side-title victim-title">VICTIM NETWORK</text>
                <text x="280" y="66" textAnchor="middle" className="graph-group-label">FOLLOWING</text>
                <text x="280" y="615" textAnchor="middle" className="graph-group-label">FOLLOWERS</text>

                <text x="590" y="36" textAnchor="middle" className="graph-middle-title">SOCIAL BRIDGE</text>

                <text x="900" y="36" textAnchor="middle" className="graph-side-title volunteer-title">VOLUNTEER NETWORK</text>
                <text x="900" y="66" textAnchor="middle" className="graph-group-label">FOLLOWING</text>
                <text x="900" y="615" textAnchor="middle" className="graph-group-label">FOLLOWERS</text>

                {/* Soft zone separators */}
                <line x1="455" y1="78" x2="455" y2="590" className="zone-divider" />
                <line x1="725" y1="78" x2="725" y2="590" className="zone-divider" />

                {/* Local fan edges first */}
                {layout.edges.filter((edge) => edge.kind === "local").map((edge, index) => {
                  const source = nodesById.get(edge.source);
                  const target = nodesById.get(edge.target);
                  if (!source || !target) return null;
                  return (
                    <path
                      key={`local-${index}`}
                      d={edgeCurve(source, target, "local")}
                      className="edge-local"
                    >
                      <title>{edge.label}</title>
                    </path>
                  );
                })}

                {/* Only real left-to-right cross-network bridge edges */}
                {layout.edges.filter((edge) => edge.kind === "cross").map((edge, index) => {
                  const source = nodesById.get(edge.source);
                  const target = nodesById.get(edge.target);
                  if (!source || !target) return null;
                  return (
                    <path
                      key={`cross-${index}`}
                      d={edgeCurve(source, target, "cross")}
                      className="edge-cross"
                    >
                      <title>{edge.label}</title>
                    </path>
                  );
                })}

                {/* Selected best path on top */}
                {layout.edges.filter((edge) => edge.kind === "path").map((edge, index) => {
                  const source = nodesById.get(edge.source);
                  const target = nodesById.get(edge.target);
                  if (!source || !target) return null;
                  return (
                    <path
                      key={`path-${index}`}
                      d={edgeCurve(source, target, "path")}
                      className="edge-selected"
                    >
                      <title>Best selected path</title>
                    </path>
                  );
                })}

                {/* Nodes last */}
                {layout.nodes.map((node) => {
                  const selected = selectedNode?.id === node.id;
                  const anchor = node.id === layout.victimId || node.id === layout.volunteerId;
                  const radius = anchor ? 20 : 11;

                  return (
                    <g
                      key={node.id}
                      transform={`translate(${node.x}, ${node.y})`}
                      className="readable-node"
                      onClick={() => setSelectedNode(node)}
                    >
                      <circle
                        r={selected ? radius + 3 : radius}
                        fill={nodeColor(node)}
                        stroke={selected ? "#0f172a" : "#fff"}
                        strokeWidth={selected ? 3 : 2}
                      />
                      <text y={radius + 18} textAnchor="middle" className="readable-node-label">
                        {shortLabel(node)}
                      </text>
                      {anchor && (
                        <text y={radius + 32} textAnchor="middle" className="anchor-role-label">
                          {node.id === layout.victimId ? "Victim" : "Volunteer"}
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            </div>

            <aside className="sna-inspector">
              {!selectedNode ? (
                <p className="sna-inspector-empty">Click a node to inspect it.</p>
              ) : (
                <>
                  <p className="eyebrow">{selectedNode.role}</p>
                  <h3>@{selectedNode.handle || selectedNode.did}</h3>

                  {selectedNode.role === "Victim" && (
                    <>
                      <p className="sna-score-breakdown">{selectedNode.score_breakdown || "Origin node."}</p>
                      <div className="sna-inspector-section">
                        <p className="eyebrow">Original Post</p>
                        <p className="sna-post-text">{selectedNode.post_text || "No post text available."}</p>
                      </div>
                      <div className="sna-inspector-section">
                        <p className="eyebrow">Specific Needs</p>
                        <p className="sna-post-text">{formatList(selectedNode.items || selectedNode.needed_items)}</p>
                      </div>
                    </>
                  )}

                  {selectedNode.role === "Volunteer" && (
                    <>
                      <p className="sna-score-breakdown">{selectedNode.score_breakdown || "Matched volunteer."}</p>
                      <div className="sna-inspector-section">
                        <p className="eyebrow">Original Post</p>
                        <p className="sna-post-text">{selectedNode.post_text || "No post text available."}</p>
                      </div>
                      <div className="sna-inspector-section">
                        <p className="eyebrow">Items Offered</p>
                        <p className="sna-post-text">{formatList(selectedNode.items || selectedNode.offered_items)}</p>
                      </div>
                    </>
                  )}

                  {selectedNode.role !== "Victim" && selectedNode.role !== "Volunteer" && (
                    <>
                      <p className="sna-score-breakdown">{selectedNode.relationship || "Network contact."}</p>
                      <div className="sna-inspector-section">
                        <p className="eyebrow">Post</p>
                        <p className="sna-post-text">{selectedNode.post_text || "No post text available."}</p>
                      </div>
                    </>
                  )}
                </>
              )}
            </aside>
          </div>
        )}

        <p className="sna-modal-hint">
          Read from top to bottom: following nodes are above each main account, followers are below. Light-gray curves belong to each local network, blue curves are real cross-network bridges, and orange shows the selected best path.
        </p>
      </section>
    </div>
  );
}
