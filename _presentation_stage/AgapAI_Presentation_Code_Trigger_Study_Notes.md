# AgapAI Spatiotemporal Pipeline: Code Triggers by Presentation Slide

These notes follow the 14-slide deck **AgapAI_Spatiotemporal_Event_Clustering_Presentation_Layout.pptx**. Each slide is mapped to the functions that produce its information and the conditions that cause those functions or branches to run.

Code paths below are relative to `agapai/api/agapai_api/`, unless a dashboard path is shown. Function line numbers refer to the reviewed code snapshot and can shift after edits.

## First, the full trigger path

1. The dashboard calls `loadSpatiotemporalData(posts)` in `agapai/dashboard/src/spatiotemporal.js:22`.
2. It sends selected post IDs to `POST /ner/process-all?limit=100` until the API reports no selected posts left to process.
3. `process_all_posts` in `ner_routes.py:375` selects stale or unprocessed records and calls `_process_post` for each one.
4. `_process_post` calls location resolution, date/time extraction, and raw entity extraction, then returns values to save in MongoDB.
5. After processing, the dashboard requests `POST /ner/results`. `get_selected_results` calls `_build_event_lookup`, which groups eligible posts and returns rows plus event summaries.
6. The dashboard table and map display the response. The browser formats returned fields; it does not rerun the Python NER or clustering algorithms.

The calls are sequential: the dashboard starts processing, the API saves post-level results, then the result endpoint creates event groups from saved results.

## Slide 1 — Title: Spatiotemporal Pipeline

**Code trigger:** None. This is the presentation title.

**What the pipeline covers:** finding and validating locations, extracting an event date/time using the post text and post timestamp, and grouping eligible located posts into events.

## Slide 2 — Pipeline flow: from posts to events

**Dashboard trigger — `loadSpatiotemporalData`**

- The UI calls `loadSpatiotemporalData` when it needs spatiotemporal information for loaded posts.
- The function builds a unique list of selected post IDs. If the list is empty, it returns empty rows and events without calling the API.
- Otherwise it repeatedly calls `process-all?limit=100`. A returned `remaining` count above zero triggers another batch. When processing is done, it calls `results`.
- Source: `agapai/dashboard/src/spatiotemporal.js:22`.

**API trigger — `process_all_posts`**

- Runs for `POST /ner/process-all` when MongoDB is connected.
- It scopes the query to the selected IDs and finds posts that are unprocessed, produced by another algorithm version, missing snapshots, or whose text/created-at no longer matches the saved snapshot.
- It processes at most the requested `limit`, capped at 100.
- Source: `ner_routes.py:84` (`_needs_ner_processing_query`) and `ner_routes.py:375` (`process_all_posts`).

**Event trigger — `_build_event_lookup`**

- Runs when the result endpoint needs event summaries.
- It selects disaster posts with at least one resolved location and a valid `created_at`, builds cluster inputs, and calls `cluster_posts_into_events`.
- Source: `ner_routes.py:445`.

## Slide 3 — Inputs and outputs

**Input selection — `_selection_query`, `_scoped_query`**

- `_selection_query` turns `post_ids` into a MongoDB `_id` filter. It supports string Bluesky URIs and valid BSON ObjectIds.
- `_scoped_query` combines the selection filter with the processing or event query.
- Source: `ner_routes.py:32` and `ner_routes.py:40`.

**Fields read — `PROCESSING_PROJECTION` and `EVENT_PROJECTION`**

- Processing reads the text, `created_at`, author DID/handle, and saved input snapshots.
- Event building reads the disaster type, resolved locations, posted time, and saved event date/time fields.
- Source: `ner_routes.py:42` and `ner_routes.py:55`.

**Saved output — `_process_post` and `process_all_posts`**

- `_process_post` returns location results, ambiguities, date/time values, raw entities, and `NER_ALGORITHM_VERSION`.
- `process_all_posts` adds `ner_processed_at`, `ner_text_snapshot`, and `ner_created_at_snapshot`, then stores the values with `posts_col.update_one`.
- Those snapshots let `_needs_ner_processing_query` detect changed inputs later.
- Source: `ner_routes.py:247` and `ner_routes.py:430`.

## Slide 4 — Location 1: find place candidates

**Model loader — `get_nlp`**

- The first request needing NLP calls `get_nlp`. If `_nlp` is empty, the function downloads/loads the configured spaCy model and caches it; later calls reuse it.
- Source: `ner_algorithm.py:30`.

**Candidate extraction — `extract_location_entities`**

- Called by `extract_and_resolve` for each text source that is being checked.
- Starts with spaCy entities labelled `LOC` or `ORG`.
- It also groups tokens using POS and dependency relations, to catch address-like text that the entity model may miss.
- Those relation labels describe sentence structure. They are not words such as city names or temporal phrases.
- `names_in_entity` then checks candidate spans against gazetteer names. A model candidate by itself is not yet a final location.
- Source: `ner_algorithm.py:40` and `ner_gazetteer_loader.py:48`.

## Slide 5 — Location 2: check place names

**Gazetteer setup — `_load_gazetteer`**

- The first lookup calls the cached loader, which reads `Cebu_Province_PSGC_Gazetteer.geojson` and builds a name-to-record index.
- Each record supplies the place type, PSGC code, parent information, and coordinates. City/municipality variants are derived from official metadata.
- Source: `ner_gazetteer_loader.py:18`.

**Name matching — `_normalize`, `find_exact`, `names_in_entity`**

- `_normalize` case-folds names, turns hyphens into spaces, and normalizes whitespace.
- `find_exact` looks up that normalized name and prefers an official non-barangay record when an official name and alias share a key.
- `names_in_entity` tries the whole NLP candidate first, then the longest matching gazetteer phrases within that candidate.
- Source: `ner_gazetteer_loader.py:13`, `ner_gazetteer_loader.py:40`, and `ner_gazetteer_loader.py:48`.

**Hierarchy and ambiguity — `resolve_location_text`, `find_by_parent`, `_location_result`**

- `resolve_location_text` uses a parent hint when it has one and returns a resolved entry only when one match remains.
- `find_by_parent` narrows matches to the stated municipality when possible; if that yields no match, it keeps the original matches.
- Multiple remaining entries become an ambiguous result. `_location_result` labels the full result `RESOLVED`, `AMBIGUOUS_REVIEW_QUEUE`, or `LOW_CONFIDENCE_REVIEW_QUEUE`.
- Source: `ner_algorithm.py:105`, `ner_gazetteer_loader.py:69`, and `ner_algorithm.py:133`.

## Slide 6 — Location 3: use author context

**Trigger order — `_resolve_with_author_context`**

1. Calls `resolve_post_location` for the post text.
2. If the result has either a resolved location or an ambiguous location, it returns. The bio is not used to replace that post result.
3. If the post returns no location and the author is known, `_fetch_author_bio` retrieves the profile bio and calls location resolution again.
4. If the bio also returns no location or ambiguity, `_fetch_previous_post_texts` retrieves up to ten earlier authored posts and retries resolution.

The feed helper keeps posts strictly earlier than the target, from the same author, and excludes duplicates and the target post. Empty text still counts toward the ten posts. API failures stop history retrieval and leave earlier processing results usable.

**Confidence assignment — `resolve_post_location`**

- Direct post-text locations default to HIGH confidence.
- Bio locations are tagged MEDIUM with source `author_bio`.
- History locations are tagged LOW with source `author_previous_post`.
- A post-text ambiguity stops the source fallback. It can still be refined by the centroid step described on slide 7.
- Source: `ner_routes.py:223`, `ner_routes.py:176`, `ner_routes.py:181`, and `ner_algorithm.py:179`.

## Slide 7 — Location 4: check ambiguous places

**Two-pass trigger — `process_all_posts`**

- Pass 1 processes the batch without the new batch's anchor set, then collects direct HIGH-confidence gazetteer locations and their post timestamps.
- Pass 2 runs only for posts whose first-pass `ner_ambiguous_locations` is non-empty.
- `_nearby_points` supplies direct anchors no more than `MAX_TIME_HOURS` (6 hours) from the target's posted timestamp.
- Source: `ner_routes.py:393`, `ner_routes.py:413`, `ner_routes.py:307`, and `ner_routes.py:347`.

**Candidate selection — `resolve_via_spatiotemporal_centroid`**

- Averages nearby anchor latitude and longitude.
- Measures each ambiguous candidate's distance to that mean using `haversine_km`.
- Returns the nearest candidate only when it is at most `SPATIOTEMPORAL_MAX_DISTANCE_KM` (5 km) away. Otherwise the result remains ambiguous.
- This 5 km check resolves one post's place candidate. The event grouping rule on slides 10–11 uses a separate 2 km pair threshold.
- Source: `ner_algorithm.py:170` and `ner_event_clustering.py:9`.

**Implementation note:** the centroid call passes `method="spatiotemporal_centroid"` to `_location` but does not override its default confidence/source values. In the current code, use `resolution_method` to identify a centroid result; its confidence fields may still carry the defaults.

## Slide 8 — Date/time 1: parse the text

**Reference timestamp — `_parse_created_at` and `extract_explicit_datetime`**

- `_parse_created_at` parses the post metadata timestamp and treats a naive Mongo datetime as UTC.
- `extract_explicit_datetime` converts the reference timestamp to `Asia/Manila` and calls `extract_vague_datetime` with the post text.
- Source: `ner_routes.py:97` and `ner_algorithm.py:345`.

**Original text parser — `extract_vague_datetime`**

- Calls `_recognized_datetime_candidates` and `_dateparser_candidates` on the original text.
- The recognizer uses `Culture.English`; dateparser searches and parses spans using the Manila timezone and metadata reference.
- A unique recognized clock result is needed. Ambiguous time alternatives are skipped rather than guessed.
- Source: `ner_algorithm.py:271`, `ner_algorithm.py:230`, and `ner_algorithm.py:319`.

**Translation branch — `_translate_temporal_text`**

- When the original text has no explicit time/datetime result, the function calls the local language-detection/translation helpers, then runs the same parsers on the translated text.
- A vague date in the original may still be retained, while an explicit translated result can take priority.
- There is no handwritten list of Filipino/Cebuano date words in this path. Parsing depends on the translation and date libraries recognizing the expression.
- Source: `ner_algorithm.py:311` and `ner_algorithm.py:319`.

## Slide 9 — Date/time 2: fill missing values

**Component completion — `_format_event_datetime`**

- Parser functions set `has_date` and `has_time` based on what the library recognized.
- If the text supplies a date but no time, `_format_event_datetime` copies the metadata time.
- If the text supplies a time but no date, it copies the metadata date.
- If both are present, it keeps both parsed components.
- Relative dates use the posted timestamp as the reference, so a recognized “yesterday” relative to a June 26 Manila timestamp resolves to June 25.
- Source: `ner_algorithm.py:206`, `ner_algorithm.py:230`, and `ner_algorithm.py:271`.

**Fallback branch — `extract_explicit_datetime`**

- If no text candidate is returned, it returns the Manila-local metadata timestamp with source `fallback`.
- If the text is unrecognized and the metadata timestamp is also absent, it returns `unresolved` with null date/time values.
- Source: `ner_algorithm.py:345`.

**Saved/display fields — `_to_ner_datetime`, `_process_post`, `_table_row`**

- `ner_datetime` stores extraction provenance and phrase details or a readable fallback timestamp.
- `ner_datetime_source` stores `explicit`, `vague_date`, `fallback`, or `unresolved`.
- `ner_event_datetime` stores the datetime value; `ner_event_date` and `ner_event_time` store the Manila-local display components.
- `_table_row` uses saved event components first, then stored event datetime, then posted metadata for table display. It does not rerun the text parser.
- Source: `ner_routes.py:112`, `ner_routes.py:247`, and `ner_routes.py:532`.

## Slide 10 — Event clustering 1: compare posts

**Eligible post filter — `_build_event_lookup`**

- A post enters the clustering input only if its disaster type is in `DISASTER_TYPES`, it has at least one resolved `ner_location`, and its `created_at` parses.
- The function selects one representative location per post with `_location_rank`.
- Ambiguous-only posts have no resolved location and are excluded at this step.
- Source: `ner_routes.py:301` and `ner_routes.py:445`.

**Pair rule — `is_same_event`**

- First, disaster types must match.
- Next, `haversine_km` must be at most `MAX_DISTANCE_KM` (2 km).
- Finally, the absolute difference between the two posts' metadata `created_at` values must be at most `MAX_TIME_HOURS` (6 hours).
- All three checks must pass. Extracted text date/time does not determine pair eligibility; `created_at` does.
- Source: `ner_event_clustering.py:5`, `ner_event_clustering.py:9`, and `ner_event_clustering.py:18`.

## Slide 11 — Event clustering 2: group posts

**Pair matrix — `build_distance_matrix`**

- Calls `is_same_event` for each pair.
- Stores `0.0` for a compatible pair and `1.0` for an incompatible pair. The matrix values are binary; they are not kilometers or hours.
- Source: `ner_event_clustering.py:36`.

**Grouping — `cluster_posts_into_events`**

- Runs DBSCAN with a precomputed matrix, `eps=0.5`, and `min_samples=1`.
- A zero-valued link joins compatible posts. Since compatible links connect transitively, A–B and B–C can form one group even when A–C does not directly pass the 2 km / six-hour checks.
- `min_samples=1` allows every eligible post to appear, including as a single-post group.
- Event labels such as `event-0` are assigned while building the API response; they are not persistent database event IDs.
- Source: `ner_event_clustering.py:48` and `ner_routes.py:476`.

## Slide 12 — Event summary: choose place and time

**Summary trigger — `build_event_summary`**

- Called once for each cluster from `_build_event_lookup`.
- Calls `resolve_event_datetime` and `pick_event_location`, then adds disaster type, member posts, and distinct author handles.
- Source: `ner_event_clustering.py:102`.

**Event time — `resolve_event_datetime`**

- Looks for cluster members with source `explicit` or `vague_date` and a non-null event datetime.
- Uses the earliest such datetime if any exist. Otherwise it uses the earliest metadata `created_at` in the group.
- Unresolved members may be assigned an inferred/fallback time in the in-memory event response. This does not update their MongoDB records.
- Source: `ner_event_clustering.py:63`.

**Representative place — `pick_event_location`**

- Chooses the member with the highest location confidence and returns that member's place and coordinates. It does not average cluster locations.
- Source: `ner_event_clustering.py:90`.

## Slide 13 — API and saved results

**Process endpoint — `process_all_posts`**

- Route: `POST /ner/process-all`.
- It processes selected posts in batches, updates MongoDB fields, and returns counts for processed and remaining posts.
- A version mismatch, missing snapshot, or changed text/created-at makes a record eligible for processing again.
- Source: `ner_routes.py:84` and `ner_routes.py:374`.

**Results endpoint — `get_selected_results`**

- Route: `POST /ner/results`.
- Calls `_build_event_lookup` for the selected IDs, reads selected table rows, attaches event IDs, and returns rows with event summaries.
- Source: `ner_routes.py:548`.

**Other display endpoints**

- `GET /ner/table-data` returns paginated rows and event IDs via `_table_row` (`ner_routes.py:562`).
- `GET /ner/events` returns event summaries via `_build_event_lookup` (`ner_routes.py:587`).

## Slide 14 — Dashboard output and limits

**Table location display — `locationDisplayItems`**

- Formats resolved locations using their breadcrumbs and matched names.
- Adds unresolved candidates as “location unresolved” display entries rather than treating them as confirmed.
- Source: `agapai/dashboard/src/spatiotemporal.js:3`.

**Table date/time display — `getStrictDateTime`, `getEventDateTime`**

- Use saved `ner_event_date` / `ner_event_time` first.
- If these are absent, read the normalized event datetime or posted `created_at` to populate the table.
- These functions format the API response; they do not run date recognition.
- Source: `agapai/dashboard/src/NERTableView.jsx:19` and `agapai/dashboard/src/NERTableView.jsx:28`.

**Map display — `NERMapView`**

- Receives event summaries and renders their coordinates, disaster marker, and popup information.
- It does not choose a gazetteer place or calculate the clusters; those decisions have already happened in the API.
- Source: `agapai/dashboard/src/NERMapView.jsx`.

**What the displayed limits mean**

- The location and date/time parsers can fail to recognize a mention. The API retains fallback or ambiguous status where available.
- Author bio/history requires Bluesky context retrieval; an unavailable response can leave that fallback empty.
- DBSCAN links compatible pairs transitively, so the spread of a complete group may exceed the direct pair thresholds.
- These behaviors come from `_resolve_with_author_context`, `extract_vague_datetime`, `extract_explicit_datetime`, and `cluster_posts_into_events`.

## Quick function lookup for the panel

| Information shown | Main trigger/function | Main output |
|---|---|---|
| Possible locations in a post | `extract_location_entities` | NLP candidate spans |
| Verified place and coordinates | `extract_and_resolve` → `resolve_location_text` → gazetteer lookup | PSGC match or ambiguity candidates |
| Bio/history location fallback | `_resolve_with_author_context` → `resolve_post_location` | Medium/low confidence location |
| Nearby-anchor disambiguation | `process_all_posts` pass 2 → `resolve_via_spatiotemporal_centroid` | Nearest candidate or retained ambiguity |
| Text date/time | `extract_explicit_datetime` → `extract_vague_datetime` | Extracted datetime, phrase, source |
| Missing component or metadata fallback | `_format_event_datetime` / `extract_explicit_datetime` | Manila-local event date/time or unresolved |
| Pairwise event compatibility | `is_same_event` | True/false using type, 2 km, 6 hours |
| Event groups | `build_distance_matrix` → `cluster_posts_into_events` | List of post clusters |
| Event summary | `build_event_summary` | Representative place, time, members |
