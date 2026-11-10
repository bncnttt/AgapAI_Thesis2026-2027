from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

PPTX = Path(__file__).parent / "AgapAI_Spatiotemporal_Event_Clustering_Presentation_Layout.pptx"
TEMP = PPTX.with_suffix(".revised.tmp")
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}
ET.register_namespace("a", NS["a"])
ET.register_namespace("p", NS["p"])

REPLACEMENTS = {
    "Spatiotemporal Intelligence": "Spatiotemporal Pipeline",
    "From already collected social posts to interpretable incident groups": "How collected posts become location and event information",
    "MAP PIN     →     CLOCK     →     CONNECTED POSTS": "PLACE     →     EVENT TIME     →     RELATED POSTS",

    "END-TO-END FLOW": "PIPELINE FLOW",
    "Pipeline at a glance": "From posts to events",
    "•  Loaded MongoDB posts → selected post IDs": "•  Select posts already stored in MongoDB",
    "•  Location and date/time extraction → NER fields saved to MongoDB": "•  Extract location and date/time; save results",
    "•  Eligible located posts → event clusters": "•  Group eligible posts into events",
    "•  API results → All Posts table, Clustered Events list, and map": "•  Show results in the table, event list, and map",
    "Draw six connected boxes: Posts → Extract → Validate → Cluster → API → Dashboard. Color location, time, and clustering separately.": "Diagram: Posts → Extract → Check → Group → Dashboard.",
    "This component does not collect or classify posts. It operates on records already loaded by AgapAI.": "This pipeline works on posts already collected by AgapAI.",

    "WHAT THIS PIPELINE OWNS": "INPUTS AND OUTPUTS",
    "Component boundary and inputs": "What this pipeline uses",
    "•  Inputs: _id, text, created_at, author DID/handle, disaster_type": "•  Inputs: post text, posted time, author, disaster type",
    "•  MongoDB posts are processed in batches of up to 100": "•  Processes up to 100 posts per batch",
    "•  Outputs include algorithm version and input snapshots": "•  Saves results and snapshots of the input text and time",
    "•  Shared dependencies: Bluesky context client, Mongo posts collection, disaster types": "•  Uses MongoDB, Bluesky author context, and disaster labels",
    "Input/output split: existing collection + classification → this pipeline → dashboard.": "Diagram: Existing posts and labels → this pipeline → dashboard.",
    "Post text and published time are primary evidence. Author identity is needed only for optional location context.": "Post text and posted time are the main evidence. Author details support location fallback.",

    "MODEL + TEXT STRUCTURE": "LOCATION 1 · FIND PLACE CANDIDATES",
    "Location 1 · Generate candidates": "Find possible place names",
    "•  Lazy-load spaCy model tl_calamancy_md (revision 0.2.0)": "•  spaCy model: tl_calamancy_md (revision 0.2.0)",
    "•  Collect LOC/ORG entities plus syntax-based address spans": "•  Finds named places and address-like text",
    "•  Syntax spans help with informal or lowercase addresses": "•  Sentence structure helps with informal addresses",
    "•  Candidates are not final places until validated by the Cebu gazetteer": "•  The Cebu gazetteer checks each candidate name",
    "Show a sample sentence with a model-highlighted phrase, then label it Candidate → Not yet verified.": "Highlight a place candidate and label it “needs checking.”",
    "The NER model proposes where to look; it does not supply final coordinates or PSGC identity.": "spaCy suggests place mentions. The gazetteer supplies the verified place and coordinates.",

    "PSGC NAME → PLACE RECORD": "LOCATION 2 · CHECK PLACE NAMES",
    "Location 2 · Gazetteer validation": "Match names to Cebu places",
    "•  GeoJSON provides official names, types, parent links, coordinates, and PSGC codes": "•  Gazetteer stores names, place types, parent links, coordinates, and PSGC codes",
    "•  Names are normalized; aliases are derived from place metadata": "•  Names are normalized; aliases come from place metadata",
    "•  Municipality hints can narrow barangay or homonym matches": "•  Municipality context can narrow a place-name match",
    "•  One match resolves; multiple matches stay explicitly ambiguous": "•  One match is resolved; multiple matches stay marked ambiguous",
    "Funnel: model phrase → gazetteer lookup → resolved record or ambiguity candidates. Show breadcrumb + PSGC code on success.": "Flow: Candidate name → gazetteer → resolved place or ambiguous matches.",
    "Gazetteer validation grounds outputs in official Cebu place records and preserves ambiguity.": "The gazetteer checks names against Cebu place records and keeps uncertain matches visible.",

    "STRONGEST AVAILABLE SOURCE": "LOCATION 3 · USE AUTHOR CONTEXT",
    "Location 3 · Evidence cascade": "Use location sources in order",
    "•  Post text first: HIGH / post_text": "•  Post text: HIGH confidence",
    "•  If no resolved or ambiguous place: author bio, MEDIUM / author_bio": "•  If no post result: author bio, MEDIUM confidence",
    "•  If bio gives no result: up to 10 preceding authored posts, LOW / author_previous_post": "•  If bio has no result: up to 10 earlier posts, LOW confidence",
    "•  An ambiguous post location is not silently replaced with the author's likely home": "•  An ambiguous post location stays for review",
    "Three-level ladder HIGH → MEDIUM → LOW. Add an early-stop marker when a valid or ambiguous result appears.": "Diagram: Post → Bio → Earlier posts. Stop when the post has a result.",
    "Bio and history are fallbacks for posts with no location evidence, not votes that override the post.": "Bio and earlier posts are checked only when the post has no location result.",

    "AMBIGUOUS PLACE + NEARBY ANCHORS": "LOCATION 4 · CHECK AMBIGUOUS PLACES",
    "Location 4 · Spatiotemporal disambiguation": "Use nearby confirmed locations",
    "•  Pass 1 collects direct HIGH-confidence gazetteer anchors": "•  First, collect direct high-confidence locations",
    "•  Pass 2 uses anchors no more than 6 hours from the target post": "•  Use anchors posted within 6 hours",
    "•  Average anchor coordinates; accept nearest ambiguous candidate only within 5 km": "•  Average anchor coordinates; choose a candidate within 5 km",
    "•  The 5 km inference rule is separate from the 2 km event-clustering rule": "•  Location check: 5 km. Event grouping: 2 km.",
    "Map sketch: anchor pins → mean point → candidate places; show 5 km acceptance radius and 6-hour window.": "Map: confirmed locations → average point → nearest candidate.",
    "If no anchor is near enough, the location stays ambiguous rather than being forced.": "If no candidate is close enough, keep the location ambiguous.",

    "RELATIVE TO PUBLICATION TIME": "DATE AND TIME 1 · PARSE THE TEXT",
    "Date/time 1 · Parse the post": "Read date and time from text",
    "•  Parse created_at; use Asia/Manila as reference and display timezone": "•  Use the post time in Asia/Manila as the reference",
    "•  Try Recognizers-Text and dateparser on original text": "•  Parse the original text with date/time libraries",
    "•  If needed, detect language and translate locally with NLLB-200, then parse again": "•  If needed, translate locally and parse again",
    "•  No manual phrase list such as kahapon is embedded in the pipeline": "•  No hand-written list of time phrases is used",
    "Original text → English parser → local translation if needed → parser result / metadata fallback.": "Flow: Text → parser → local translation if needed → parser → metadata fallback.",
    "Relative phrases use the post timestamp as their reference. Mixed-language text can still challenge language detection and parsing.": "Relative dates use the post time as a reference. Translation and parsing may miss a phrase.",

    "TEXT FIRST, METADATA FOR GAPS": "DATE AND TIME 2 · FILL MISSING VALUES",
    "Date/time 2 · Complete missing pieces": "Use metadata for missing parts",
    "•  Full date + time in text → use both extracted components": "•  Date and time in text → use both",
    "•  Date only → extracted date + metadata post time": "•  Date only → add the posted time",
    "•  Time only → metadata post date + extracted time": "•  Time only → add the posted date",
    "•  No parseable indicator → metadata date/time; no usable metadata → unresolved": "•  No parse → use posted date and time; otherwise unresolved",
    "2×2 date/time grid. Example: June 26, 2026 PHT + recognized ‘yesterday’ → June 25; time stays the posted time.": "Example: recognized “yesterday” + a June 26 post time → June 25; keep the posted time.",
    "The code does not subtract one day for a literal word; the translation/parser must recognize it.": "The parser must recognize “yesterday” before it can set the date to June 25.",

    "WHEN TWO POSTS CAN RELATE": "EVENT CLUSTERING 1 · COMPARE POSTS",
    "Clustering 1 · Candidate posts and rule": "Rules for linking posts",
    "•  Candidate needs allowed disaster_type, resolved location, and valid created_at": "•  Post needs a disaster label, resolved location, and posted time",
    "•  Disaster types must match": "•  Disaster types must match",
    "•  Haversine distance must be ≤ 2 km": "•  Locations must be within 2 km",
    "•  Metadata created_at difference must be ≤ 6 hours": "•  Posted times must be within 6 hours",
    "Pairwise decision card with type match, 2 km radius, and 6-hour timeline as AND gates. Ambiguous-only posts do not enter clusters.": "Diagram: Same type AND within 2 km AND within 6 hours. Ambiguous-only posts are excluded.",
    "Membership uses the posts' metadata creation times, not a date inferred from wording.": "The six-hour check uses posted times. Text dates are used in the event summary.",

    "CONNECTED COMPATIBLE POSTS": "EVENT CLUSTERING 2 · GROUP POSTS",
    "Clustering 2 · DBSCAN groups": "Group linked posts",
    "•  Matrix: compatible pair = 0; incompatible pair = 1": "•  Pair matrix: match = 0; no match = 1",
    "•  DBSCAN: precomputed distance, eps 0.5, min_samples 1": "•  DBSCAN groups matching pairs",
    "•  Compatible chains can join transitively; eligible posts can be singleton clusters": "•  A–B and B–C can join even if A–C does not match",
    "•  event-0, etc. are response IDs, not permanent database IDs": "•  Each eligible post can form its own group; IDs are temporary",
    "Graph: A—B and B—C are compatible, A—C is not; show one connected cluster and a separate singleton.": "Diagram: A–B and B–C link; A–C does not. The links can still form one group.",
    "Thresholds define graph edges, not a maximum diameter for the entire cluster.": "The thresholds apply to each link; a full group can span farther.",

    "ONE REPRESENTATIVE SUMMARY PER GROUP": "EVENT SUMMARY",
    "Event summary · Place and time": "Choose place and time",
    "•  Representative place comes from the highest-confidence member location": "•  Place comes from the member with highest location confidence",
    "•  If extracted explicit/vague event times exist, choose the earliest": "•  Use the earliest extracted event time, when available",
    "•  Otherwise use the earliest metadata created_at in the group": "•  Otherwise use the earliest posted time",
    "•  Extracted event date affects summary display, not cluster membership": "•  Text date affects the summary, not group membership",
    "Several posts flow into a summary card: type, place/coordinates, date/time, post count, authors.": "Show a summary with type, place, date/time, and post count.",
    "A date-only phrase can anchor the event date; its time-of-day may come from post metadata.": "A date-only phrase can use the posted time to fill the missing time.",

    "REPROCESS SAFELY, RETURN CONSISTENT RESULTS": "API AND SAVED RESULTS",
    "API and persistence": "Save and return results",
    "•  POST /ner/process-all?limit=100 processes selected IDs in batches": "•  /ner/process-all?limit=100 processes selected posts",
    "•  Saves locations, ambiguities, date/time, provenance, and snapshots": "•  Saves locations, date/time, sources, and input snapshots",
    "•  POST /ner/results returns selected rows and events together": "•  /ner/results returns post rows and event groups",
    "•  Version/text/timestamp changes trigger reprocessing; unchanged records are skipped": "•  Changed text or post time triggers reprocessing",
    "Sequence: Dashboard → process-all → MongoDB updates → results → dashboard state.": "Flow: Dashboard → process-all → MongoDB → results → dashboard.",
    "Version and snapshots prevent stale NER output from surviving changed text or timestamps.": "The saved version and snapshots help detect changed posts.",

    "READABLE OUTPUT + HONEST LIMITS": "DASHBOARD OUTPUT",
    "Dashboard and evaluation": "Display results and limits",
    "•  All Posts: locations, confidence, ambiguity, PHT date and time": "•  All Posts shows locations, confidence, ambiguity, date, and time",
    "•  Clustered Events: summaries and members; map: event markers and popups": "•  Events view shows groups; the map shows event locations",
    "•  Models can miss phrases; metadata and ambiguity fallbacks remain visible": "•  Parsing can miss phrases; fallback and ambiguous results are shown",
    "•  Live proof should show a real post and actual bio/history or parser provenance": "•  Show a real post and its saved location and date/time sources",
    "Use table, cluster-card, and map screenshots. Add limits: model errors, network context fetch, chain clustering.": "Show one table row, one event group, and one map result.",
    "Show confidence source and resolution method. A synthetic test alone does not prove live Bluesky retrieval.": "Point out the location source, date/time source, and any remaining ambiguity.",
}


def replace_paragraph_text(paragraph, value):
    text_nodes = paragraph.findall(".//a:t", NS)
    if not text_nodes:
        return
    text_nodes[0].text = value
    for node in text_nodes[1:]:
        node.text = ""


def main():
    replacements_left = set(REPLACEMENTS)
    with zipfile.ZipFile(PPTX, "r") as source, zipfile.ZipFile(TEMP, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", item.filename):
                root = ET.fromstring(payload)
                for paragraph in root.findall(".//p:sp/p:txBody/a:p", NS):
                    old_text = "".join(node.text or "" for node in paragraph.findall(".//a:t", NS))
                    if old_text in REPLACEMENTS:
                        replace_paragraph_text(paragraph, REPLACEMENTS[old_text])
                        replacements_left.discard(old_text)
                payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            target.writestr(item, payload)

    if replacements_left:
        TEMP.unlink(missing_ok=True)
        raise RuntimeError("Unmatched slide text: " + " | ".join(sorted(replacements_left)))
    TEMP.replace(PPTX)
    with zipfile.ZipFile(PPTX) as package:
        bad = package.testzip()
        if bad:
            raise RuntimeError(f"Corrupt PPTX component: {bad}")
        for item in package.namelist():
            if item.startswith("ppt/slides/slide") and item.endswith(".xml"):
                ET.fromstring(package.read(item))
    print(f"Revised {PPTX} ({PPTX.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
