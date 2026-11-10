from pathlib import Path
import re
import zipfile
from xml.etree import ElementTree as ET

PPTX = Path(__file__).parent / "AgapAI_Spatiotemporal_Detailed_Presentation.pptx"
TEMP = PPTX.with_suffix(".tmp")
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}
ET.register_namespace("a", NS["a"])
ET.register_namespace("p", NS["p"])


def slide(kicker, title, bullets, note_head_1, note_1, note_head_2, note_2):
    return {
        "kicker": kicker,
        "title": title,
        "bullets": ["•  " + bullet for bullet in bullets],
        "notes": [note_head_1, note_1, note_head_2, note_2],
    }


SLIDES = {
    2: slide(
        "PIPELINE FLOW", "From posts to event summaries",
        [
            "Select posts already collected in MongoDB.",
            "Process up to 100 posts per batch.",
            "Extract and save validated locations and event date/time.",
            "Cluster eligible located posts and return rows and events to the dashboard.",
        ],
        "KEY POINT", "Location and date/time are processed for each post before event grouping.",
        "PROCESS ORDER", "Select → extract → validate → save → cluster → display.",
    ),
    3: slide(
        "INPUT → PROCESS → OUTPUT", "How each post is processed",
        [
            "Input: post text, created_at, author identity, and upstream disaster type.",
            "Process: extract location candidates, validate them, and parse event date/time from text.",
            "Fallbacks: use bio/history only when post text has no location; use created_at for missing or unparsed time.",
            "Output: save location/time results and provenance; snapshots detect changed posts for reprocessing.",
        ],
        "LOCATION FLOW", "Post text → gazetteer → optional bio/history or nearby-anchor check.",
        "DATE/TIME FLOW", "Original text → Recognizers-Text/dateparser → translation if needed → metadata fallback.",
    ),
    4: slide(
        "LOCATION 1 · CANDIDATES", "Find possible place mentions",
        [
            "spaCy model: ljvmiranda921/tl_calamancy_md, revision 0.2.0.",
            "Collect model entities labelled LOC or ORG.",
            "Also build spans from proper nouns and address-like grammatical relations.",
            "Candidate spans still require gazetteer validation before becoming locations.",
        ],
        "GRAMMAR CUES", "Relations include oblique, noun modifier, compound, apposition, coordination, and list.",
        "IMPORTANT", "The model proposes text spans. It does not provide final PSGC identity or coordinates.",
    ),
    5: slide(
        "LOCATION 2 · GAZETTEER", "Match candidates to Cebu place records",
        [
            "Cebu PSGC gazetteer provides official names, types, parents, coordinates, and codes.",
            "Normalize case, hyphens, and whitespace before lookup.",
            "Check the full candidate; otherwise use the longest known place phrase inside it.",
            "Use municipality context to narrow matches; retain unresolved candidates as ambiguous.",
        ],
        "ALIASES", "Municipality aliases are derived from official names and administrative metadata.",
        "RESULT", "One match resolves. Multiple matches remain visible for review.",
    ),
    6: slide(
        "LOCATION 3 · AUTHOR CONTEXT", "Use location sources in order",
        [
            "Post text first: HIGH confidence.",
            "If the post has no resolved or ambiguous place, check the author bio: MEDIUM.",
            "If the bio has no result, check up to 10 earlier authored posts: LOW.",
            "Earlier posts must be by the same author and strictly before the target post.",
        ],
        "STOP RULE", "A resolved or ambiguous post-text result stops the bio/history fallback.",
        "HISTORY", "The target and duplicates are excluded; empty posts still count toward the limit of 10.",
    ),
    7: slide(
        "LOCATION 4 · SPATIAL REFINEMENT", "Resolve an ambiguous place using nearby anchors",
        [
            "First pass collects direct HIGH-confidence gazetteer locations.",
            "Second pass checks ambiguous posts against anchors within 6 hours.",
            "Average anchor coordinates; select the nearest candidate only if it is within 5 km.",
            "If no candidate qualifies, keep the location ambiguous.",
        ],
        "5 KM RULE", "This resolves one place candidate. It is separate from the 2 km event-pair rule.",
        "CONFIDENCE NOTE", "Centroid method is recorded; current code leaves confidence/source defaults unchanged.",
    ),
    8: slide(
        "DATE/TIME 1 · ORIGINAL TEXT", "Name the date and time parsers",
        [
            "Microsoft Recognizers-Text DateTime: recognizers-text-date-time 1.0.2a2.",
            "Recognizers-Text runs with Culture.English and a post-time reference.",
            "dateparser 1.4.3+: search_dates and DateDataParser parse matched expressions.",
            "Ambiguous clock resolutions are skipped rather than guessed.",
        ],
        "DATEPARSER SETTINGS", "Asia/Manila timezone, timezone-aware results, past-date preference, and relative base.",
        "RANGE RULE", "Cross-calendar-day ranges are rejected; accepted same-day ranges use that date.",
    ),
    9: slide(
        "DATE/TIME 2 · LANGUAGE FALLBACK", "Detect language and translate locally",
        [
            "Language detector: langdetect.",
            "Translation model: NLLB-200, facebook/nllb-200-distilled-600M.",
            "Translate to English locally, then run the same date/time parsers again.",
            "Translation runs when original text has no explicit time or datetime result.",
        ],
        "LANGUAGE CODES", "English → eng_Latn; Tagalog → tgl_Latn; other detections default to Cebuano.",
        "LIMIT", "langdetect has no distinct Cebuano class; short or mixed-language text can be misdetected.",
    ),
    10: slide(
        "DATE/TIME 3 · COMPLETE THE RESULT", "Use metadata for missing components",
        [
            "Full date and time recognized → use both extracted components.",
            "Date only → fill time from created_at; time only → fill date from created_at.",
            "No parseable expression → use metadata date/time; no valid metadata → unresolved.",
            "Relative phrases use the post timestamp in Asia/Manila as their reference.",
        ],
        "EXAMPLE", "If “yesterday” is recognized for a June 26 PHT post, the event date is June 25.",
        "NO PHRASE LIST", "There is no hand-written temporal vocabulary; libraries parse the original or translated text.",
    ),
    11: slide(
        "SAVED POST RESULTS", "Store values and their sources",
        [
            "ner_datetime stores provenance and phrase/resolution details or fallback information.",
            "ner_datetime_source: explicit, vague_date, fallback, or unresolved.",
            "ner_event_datetime stores the normalized instant; event_date/time are Manila-local strings.",
            "The table uses saved date/time first, then event datetime, then posted metadata.",
        ],
        "WHY STORE SOURCES", "It shows whether a result came from text, translation, or metadata fallback.",
        "TIMEZONE", "created_at is normalized to Asia/Manila for local interpretation and display.",
    ),
    12: slide(
        "CLUSTERING 1 · ELIGIBLE POSTS", "Select posts with enough information to compare",
        [
            "Disaster type must be flood, earthquake, typhoon, fire, or landslide.",
            "At least one location must be resolved to coordinates.",
            "A valid created_at timestamp is required.",
            "Choose one location per post by confidence, then breadcrumb specificity.",
        ],
        "EXCLUDED", "Posts with ambiguous-only locations or invalid posting times do not enter clustering.",
        "DATE/TIME ROLE", "Text-extracted event time helps the summary; created_at controls pair matching.",
    ),
    13: slide(
        "CLUSTERING 2 · MATCH AND GROUP", "Apply pair rules, then run DBSCAN",
        [
            "Pair match requires the same disaster type, distance ≤ 2 km, and posted-time difference ≤ 6 h.",
            "Haversine distance uses Earth radius 6,371 km.",
            "Binary matrix: compatible pair = 0; incompatible pair = 1.",
            "DBSCAN: precomputed distances, eps 0.5, min_samples 1.",
        ],
        "GROUP BEHAVIOR", "Links are transitive: A–B and B–C can join even if A–C does not match.",
        "SINGLE POSTS", "min_samples=1 allows an eligible post to form a cluster by itself.",
    ),
    14: slide(
        "EVENT SUMMARY AND DISPLAY", "Choose a representative place and time",
        [
            "Event location comes from the member with the highest location confidence; coordinates are not averaged.",
            "Event time uses the earliest explicit/vague event datetime when available.",
            "If none exists, use the earliest posted created_at timestamp.",
            "Return event summaries and post rows for the table, event list, and map.",
        ],
        "MEMBERSHIP VS SUMMARY", "Extracted dates do not change group membership; type, distance, and posted time do.",
        "EVENT IDS", "IDs such as event-0 are generated for the response, not saved as permanent event records.",
    ),
}


def set_paragraph(paragraph, text):
    nodes = paragraph.findall(".//a:t", NS)
    if not nodes:
        return
    nodes[0].text = text
    for node in nodes[1:]:
        node.text = ""


def set_shape(shape, values):
    paragraphs = shape.findall("./p:txBody/a:p", NS)
    nonempty = [p for p in paragraphs if "".join((n.text or "") for n in p.findall(".//a:t", NS)).strip()]
    if len(nonempty) != len(values):
        raise ValueError(f"Expected {len(values)} paragraphs, found {len(nonempty)}")
    for paragraph, value in zip(nonempty, values):
        set_paragraph(paragraph, value)


def main():
    with zipfile.ZipFile(PPTX, "r") as source, zipfile.ZipFile(TEMP, "w", zipfile.ZIP_DEFLATED) as target:
        for item in source.infolist():
            payload = source.read(item.filename)
            match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", item.filename)
            if match:
                slide_num = int(match.group(1))
                root = ET.fromstring(payload)
                shapes = root.findall(".//p:sp", NS)
                if slide_num == 1:
                    set_shape(shapes[2], ["AGAPAI · THESIS PRESENTATION"])
                    set_shape(shapes[3], ["Spatiotemporal Pipeline"])
                    set_shape(shapes[4], ["Location retrieval  •  Date/time retrieval  •  Event clustering"])
                    set_shape(shapes[5], [
                        "Current pipeline: extraction, validation, metadata fallback, and event grouping",
                        "Presenter: [Your name]  |  [Course / panel / date]",
                    ])
                    set_shape(shapes[6], ["POSTS  →  LOCATION + TIME  →  EVENT GROUPS"])
                elif slide_num in SLIDES:
                    content = SLIDES[slide_num]
                    set_shape(shapes[2], [content["kicker"]])
                    set_shape(shapes[3], [content["title"]])
                    set_shape(shapes[4], content["bullets"])
                    set_shape(shapes[5], content["notes"])
                payload = ET.tostring(root, encoding="utf-8", xml_declaration=True)
            target.writestr(item, payload)

    TEMP.replace(PPTX)
    with zipfile.ZipFile(PPTX) as package:
        bad = package.testzip()
        if bad:
            raise RuntimeError(f"Corrupt PPTX component: {bad}")
        slides = [n for n in package.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        if len(slides) != 14:
            raise RuntimeError(f"Expected 14 slides, found {len(slides)}")
        for name in slides:
            ET.fromstring(package.read(name))
    print(f"Created detailed 14-slide presentation: {PPTX} ({PPTX.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
