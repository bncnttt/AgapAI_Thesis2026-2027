import { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:8000/ner/details";

const LABEL_COLORS = { PER: "#fde68a", ORG: "#bfdbfe", LOC: "#bbf7d0" };

function highlightEntities(text, entities) {
  if (!entities || entities.length === 0) return text;
  const sorted = [...entities].sort((a, b) => a.start_char - b.start_char);
  const pieces = [];
  let cursor = 0;

  sorted.forEach((ent, i) => {
    if (ent.start_char > cursor) pieces.push(text.slice(cursor, ent.start_char));
    pieces.push(
      <span key={i} style={{ backgroundColor: LABEL_COLORS[ent.label] || "#e5e7eb", padding: "1px 3px", borderRadius: 3 }} title={ent.label}>
        {text.slice(ent.start_char, ent.end_char)}
      </span>
    );
    cursor = ent.end_char;
  });

  if (cursor < text.length) pieces.push(text.slice(cursor));
  return pieces;
}

export default function NERPanel({ postId, onClose }) {
  const [post, setPost] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadDetails() {
      try {
        const response = await fetch(`${API_BASE}/${postId}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to load NER details.");
        setPost(data.post);
      } catch (err) {
        setError(err.message);
      }
    }
    loadDetails();
  }, [postId]);

  return (
    <div style={{
      position: "fixed", top: 0, right: 0, width: 420, height: "100%",
      background: "#fff", borderLeft: "1px solid #ccc", padding: 20,
      overflowY: "auto", boxShadow: "-2px 0 8px rgba(0,0,0,0.15)",
    }}>
      <button onClick={onClose} style={{ float: "right" }}>Close</button>
      <h3>NER Process Breakdown</h3>

      {error && <p style={{ color: "red" }}>{error}</p>}
      {!post && !error && <p>Loading...</p>}

      {post && (
        <div>
          <h4>Original Text (highlighted)</h4>
          <p>{highlightEntities(post.text, post.ner_all_entities)}</p>

          <h4>Persons</h4>
          <p>{post.ner_persons ? post.ner_persons.join(", ") : "None/NA"}</p>

          <h4>Organizations</h4>
          <p>{post.ner_organizations ? post.ner_organizations.join(", ") : "None/NA"}</p>

          <h4>Location</h4>
          {post.ner_coordinates && !post.ner_coordinates.ambiguous ? (
            <p>
              {post.ner_coordinates.matched_name} ({post.ner_coordinates.latitude}, {post.ner_coordinates.longitude})
              <br /><em>Source: {post.ner_coordinates.source}</em>
            </p>
          ) : post.ner_coordinates?.ambiguous ? (
            <p style={{ color: "orange" }}>Ambiguous — {post.ner_coordinates.candidate_count} possible matches.</p>
          ) : (
            <p>None/NA</p>
          )}

          <h4>Date/Time</h4>
          {post.ner_datetime?.source === "extracted_from_text" ? (
            <ul>
              {post.ner_datetime.expressions.map((e, i) => (
                <li key={i}>"{e.raw_phrase}" → {e.normalized_datetime || "unresolved"}</li>
              ))}
            </ul>
          ) : (
            <p>{post.ner_datetime?.fallback_readable || "None/NA"} <em>(from post's actual posting time)</em></p>
          )}
        </div>
      )}
    </div>
  );
}