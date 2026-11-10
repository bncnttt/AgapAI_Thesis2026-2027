# Spatiotemporal and Event Clustering: Function-by-Function Study Notes

**Code reviewed:** `agapai/api/agapai_api/ner_routes.py`, `ner_algorithm.py`, `ner_gazetteer_loader.py`, `ner_event_clustering.py`, plus the spatiotemporal dashboard components.

**Scope:** location retrieval, date/time retrieval, author-context fallbacks, spatial refinement, event grouping, API data flow, and display. Line numbers refer to the reviewed source snapshot and can shift if the files change.

## 1. How to read the pipeline

The dashboard sends selected post IDs to the NER API. The API processes each post, stores location and event-date fields in MongoDB, builds an event lookup from eligible posts, and returns rows and event summaries. Location parsing, date/time parsing, and event grouping are separate stages.

Main path:

1. `spatiotemporal.js:22` calls `/ner/process-all` in batches, then calls `/ner/results`.
2. `ner_routes.py:375` processes posts and calls `_process_post` (`:247`).
3. `_process_post` calls `_resolve_with_author_context` (`:223`) for location, `extract_explicit_datetime` for date/time, and `extract_raw_entities` for diagnostic entity fields.
4. Processed fields are written to MongoDB at `ner_routes.py:430-435`.
5. `_build_event_lookup` (`ner_routes.py:445`) prepares cluster inputs and calls `cluster_posts_into_events`.
6. Event summaries and post rows are returned by `/ner/results`, `/ner/table-data`, and `/ner/events`.

## 2. Location retrieval: exact functions and responsibilities

### Location function flow

`process_all_posts` → `_process_post` → `_resolve_with_author_context` → `resolve_post_location` → `extract_and_resolve` → `extract_location_entities` → `names_in_entity` / gazetteer lookup → `_location` and `_location_result`.

### `ner_routes.py` functions

**`process_all_posts` (`ner_routes.py:374-442`)**

- Reads a selected batch from MongoDB; the batch size is limited to 100.
- Runs a first pass to get direct post-text locations and collect high-confidence location anchors.
- Runs a second pass for posts that still have ambiguous locations, using nearby anchor points.
- Saves the NER fields and input snapshots. Snapshots let the query reprocess a post if its text or posted timestamp changed.
- Once a selected batch is complete, `_refine_selected_locations` can revisit eligible selected posts with the full set of available anchors.

**`_resolve_with_author_context` (`ner_routes.py:223-245`)**

- Tries the post text first.
- If the result has either a resolved location or an ambiguous location, it returns immediately. An ambiguous location in the post prevents the bio or history from substituting a different location.
- If the post has no location result, it fetches and checks the author's bio.
- Only if the bio also has no resolved or ambiguous location does it fetch up to ten earlier authored posts.
- Caches fetched bio/history for reuse during processing.

**`_fetch_author_bio` (`ner_routes.py:176-178`)**

- Calls `get_author_bio_text` from `bluesky.py` to get the author's profile description. The algorithm treats it as context, not as the current post's direct location evidence.

**`_fetch_previous_post_texts` (`ner_routes.py:181-220`)**

- Uses the Bluesky author feed and retries only through the shared rate-limit retry helper.
- Keeps posts by the same author, strictly earlier than the current post, and excludes the current post, reposts/other authors, missing timestamps, and duplicates.
- Stops after ten earlier authored posts. Empty-text authored posts still count toward ten.
- A feed/API error is logged and history retrieval stops; the location pipeline can still use the post and bio.

**`_process_post` (`ner_routes.py:247-298`)**

- Parses `created_at`, gets or reuses a spaCy `Doc` for the post text, and invokes location resolution with that document.
- Converts resolved location objects into stored fields such as `location_text`, `matched_name`, `confidence`, `confidence_source`, `resolution_method`, coordinates, PSGC code, and breadcrumb.
- Calls `extract_explicit_datetime` with the post text and metadata timestamp as the reference.
- Returns `ner_locations`, `ner_ambiguous_locations`, date/time fields, raw entities, and the algorithm version.

**`_direct_anchor_locations` (`ner_routes.py:307-310`)**

- Selects HIGH confidence locations resolved directly by the gazetteer (`resolution_method == "gazetteer"`).
- If there are several, keeps those with the best confidence/specificity rank.
- These direct results are the evidence used by the spatiotemporal location refinement.

**`_load_saved_high_confidence_points` (`ner_routes.py:313-344`)**

- Loads saved direct anchors only when they were produced by the current `NER_ALGORITHM_VERSION` and their text and created-at snapshots still match the post.
- Parses timestamps and returns each eligible location's latitude, longitude, post ID, and timestamp.
- Excludes IDs being recalculated in the current batch so stale versions of those posts are not treated as anchors.

**`_nearby_points` (`ner_routes.py:347-356`)**

- For the post being resolved, returns other anchor coordinates within `MAX_TIME_HOURS` (six hours) of its posted metadata timestamp.
- It excludes the post itself. This function filters by time; it does not calculate the spatial mean.

**`_refine_selected_locations` (`ner_routes.py:359-371`)**

- Reloads valid high-confidence direct anchors after selected posts finish processing.
- Reprocesses selected posts that have ambiguous, medium/low-confidence, contextual, centroid-derived, or no locations.
- This allows selection order and batch boundaries to have less influence on an ambiguous location's final resolution.

### `ner_algorithm.py` functions

**`get_nlp` (`ner_algorithm.py:30-37`)**

- Downloads/loads the configured spaCy model `ljvmiranda921/tl_calamancy_md` at revision `0.2.0` on first use, then caches it in `_nlp`.
- The same model is used for tokenization, dependency structure, and entity predictions.

**`extract_location_entities` (`ner_algorithm.py:40-91`)**

- Receives the text and optionally the already-created spaCy `Doc`.
- Starts with model entities labelled `LOC` or `ORG`.
- Also forms candidate spans using proper nouns and nouns/verbs in selected dependency relations (`obl`, `nmod`, `compound`, `flat`, `appos`, `conj`, `list`). This helps with informal or lowercase address-like text.
- The relation names are grammatical labels, not a list of location words.
- It sends candidate spans to `names_in_entity`, removes contained duplicate spans, and joins adjacent matched place names when gazetteer parent/child relationships support that join.
- This function produces candidates, not final validated locations.

**`extract_raw_entities` (`ner_algorithm.py:94-102`)**

- Returns raw model entities labelled `LOC`, `PER`, and `ORG` for diagnostics or display.
- It is separate from validated locations: a raw `LOC` entity can be absent from the Cebu gazetteer and therefore not become a final location.

**`resolve_location_text` (`ner_algorithm.py:105-121`)**

- Extracts gazetteer-known place fragments from a candidate.
- Uses a second fragment, or a supplied municipality hint, to narrow the first fragment by administrative parent.
- If multiple matches remain and exactly one is a city/municipality-level settlement, it chooses that settlement; otherwise it returns an ambiguity object containing all candidate records.

**`_location` (`ner_algorithm.py:124-131`)**

- Converts a gazetteer entry into the standard location object: display text, matched name, lat/lon, PSGC code, breadcrumb, confidence, source, and resolution method.
- Defaults are HIGH / `post_text` / `gazetteer`. Callers may override confidence and source for bio or history results.
- Implementation detail: the centroid call currently sets only `method="spatiotemporal_centroid"`; its confidence/source therefore retain `_location` defaults. The method field is the reliable way to identify this path in the saved record.

**`_location_result` (`ner_algorithm.py:133-142`)**

- Deduplicates resolved locations by PSGC code and ambiguous groups by their sorted candidate PSGC codes.
- Sets status to `RESOLVED`, `AMBIGUOUS_REVIEW_QUEUE`, or `LOW_CONFIDENCE_REVIEW_QUEUE` depending on whether a resolved result or remaining ambiguity exists.

**`extract_and_resolve` (`ner_algorithm.py:145-167`)**

- Runs candidate extraction and gazetteer validation.
- Uses a resolved city/municipality mentioned elsewhere in the same text as a parent hint to narrow another ambiguous candidate.
- Produces final resolved locations and unresolved candidate groups for that source text.

**`resolve_via_spatiotemporal_centroid` (`ner_algorithm.py:170-176`)**

- A location disambiguation step, not the event clustering algorithm.
- Averages coordinates of nearby HIGH-confidence anchors, finds the closest candidate to that mean using `haversine_km`, and accepts it only within `SPATIOTEMPORAL_MAX_DISTANCE_KM` (5 km).
- Returns no result if there are no candidates/anchors or every candidate is farther than 5 km.

**`resolve_post_location` (`ner_algorithm.py:179-203`)**

- Implements the location source order: post text (HIGH), author bio (MEDIUM), then up to ten prior authored posts (LOW).
- Any resolved or ambiguous result from the post-text extraction ends this source cascade. Remaining ambiguities may be narrowed with the centroid method; the bio/history do not replace them.
- Bio/history locations are tagged with their confidence and `confidence_source`.

### `ner_gazetteer_loader.py` functions

**`_normalize` (`ner_gazetteer_loader.py:13-15`)**

- Case-folds, changes hyphens to spaces, collapses repeated whitespace, and trims the name for matching.

**`_load_gazetteer` (`ner_gazetteer_loader.py:18-38`)**

- Reads `Cebu_Province_PSGC_Gazetteer.geojson`, extracts official properties and coordinates, and builds a cached lookup.
- Derives city/municipality name variants by removing administrative words and spaCy English stop words from the official name. It does not maintain a separate manually written list of place aliases.
- The gazetteer is geographically scoped to Cebu Province.

**`find_exact` (`ner_gazetteer_loader.py:40-46`)**

- Performs normalized exact lookup. An official non-barangay name is preferred over an alias when both share a key.

**`names_in_entity` (`ner_gazetteer_loader.py:48-67`)**

- Validates an entire candidate span first; if that fails, scans its tokens and takes the longest gazetteer phrase that matches at each position.
- It searches inside an NLP candidate span, not every arbitrary substring of the whole post.

**`find_by_parent` (`ner_gazetteer_loader.py:69-76`)**

- Narrows exact-name matches to entries whose `parent_city_municipality` matches the parent hint, when possible. If narrowing finds none, it preserves the original matches so evidence is not silently discarded.

**`build_breadcrumb` (`ner_gazetteer_loader.py:78-89`)**

- Produces a four-level display path `[region, province, city/municipality, barangay]`. The source file has no region field, so region is `None`; lower-level fields depend on the entry type.

## 3. Date and time retrieval: exact functions and responsibilities

### Core rule

The post's `created_at` is parsed and converted to `Asia/Manila`, then used as the reference point for the text parser. If the post contains a parseable date but no time, the displayed event time comes from the posted metadata time. If it contains a parseable time but no date, its event date comes from the posted metadata date. If no date/time expression is parsed, both date and time fall back to the posted metadata. If metadata itself is missing or invalid, the result is unresolved.

There is no hand-maintained phrase list such as translations of “yesterday” or named clock phrases in these functions. The code uses date/time recognizers and dateparser on the original text, then uses the project's language detection and local translation model as a fallback parser input.

### Date/time function flow

`_process_post` → `_parse_created_at` → `extract_explicit_datetime` → `extract_vague_datetime` → `_recognized_datetime_candidates` and `_dateparser_candidates` → optionally `_translate_temporal_text` → `_format_event_datetime` → `_to_ner_datetime`, `_to_mongo_datetime`, and stored date/time fields.

### `ner_routes.py` functions

**`_parse_created_at` (`ner_routes.py:97-109`)**

- Accepts ISO strings (including `Z`) or Python datetimes.
- Converts a naive Mongo datetime to timezone-aware UTC; malformed/unsupported values return `None`.
- This is the metadata timestamp foundation for location-history ordering and event-date fallback.

**`_process_post` date/time block (`ner_routes.py:247-298`)**

- Uses parsed `created_at` as `reference_date`.
- Caches the time result by `(text, reference_date)`, calls `extract_explicit_datetime`, converts it to the explainable `ner_datetime` record, and returns:
  - `ner_datetime`: source plus normalized expression details or fallback display value.
  - `ner_datetime_source`: `explicit`, `vague_date`, `fallback`, or `unresolved`.
  - `ner_event_datetime`: normalized datetime value.
  - `ner_event_date`: ISO calendar date in Manila time.
  - `ner_event_time`: 12-hour time string in Manila time.

**`_to_ner_datetime` (`ner_routes.py:112-134`)**

- For `explicit` / `vague_date`, stores the original phrase and, when supplied, translated phrase, recognizer resolution, and normalized datetime.
- For `fallback`, stores the metadata timestamp in a readable `fallback_readable` field and marks `fallback_created_at`.
- For no usable value, stores `unresolved`.

**`_to_mongo_datetime` (`ner_routes.py:136-143`)**

- Leaves a datetime intact; converts a date to midnight; maps other values to `None`.
- Used to store the normalized event instant separately from the date/time strings.

**`_format_datetime_parts` (`ner_routes.py:146-167`)**

- Converts a datetime to `Asia/Manila` before formatting a date and 12-hour time.
- Treats a naive datetime as UTC, consistent with Mongo defaults.

**`_table_row` (`ner_routes.py:532-545`)**

- Fills missing table date/time values from saved event datetime or posted timestamp.
- For time, an explicit or fallback datetime can supply the time; otherwise posted metadata time is used if no stored event time is present.
- This is a display-data safeguard; it does not rerun NLP parsing.

### `ner_algorithm.py` date/time functions

**`LOCAL_TIMEZONE` (`ner_algorithm.py:21`)**

- The pipeline's local display and interpretation zone is `Asia/Manila`.
- API datetimes represent instants; `ner_event_date` and `ner_event_time` are generated in this local timezone, so the local date may differ from the UTC calendar date.

**`_format_event_datetime` (`ner_algorithm.py:206-228`)**

- Normalizes parsed datetimes into Manila time.
- When there is a metadata reference, fills the missing component:
  - `has_date=True, has_time=False`: keep parsed calendar date, copy reference time.
  - `has_date=False, has_time=True`: copy reference date, keep parsed time.
  - Both true: keep parsed date and time.
- Returns the datetime, ISO date, 12-hour time, and source label.
- A vague relative date still becomes a complete datetime because its time component is filled from metadata; that does not mean the post text explicitly stated a clock time.

**`_recognized_datetime_candidates` (`ner_algorithm.py:271-309`)**

- Calls `recognize_datetime` with `Culture.English` and a reference date.
- Requires one unique recognizer resolution. If the text leaves multiple possible clock interpretations (for example unspecified AM/PM alternatives), it skips that result instead of choosing one.
- Handles supported `time`, `date`, `datetime`, `daterange`, and `datetimerange` result types. Unsupported/malformed results are skipped.
- Rechecks time-only clock text with `DateDataParser` so an invalid clock is not accepted merely because a recognizer repaired it.
- Rejects a date range spanning more than one calendar day; accepted range results are represented at the start of that day.
- Passes date/time-presence flags to `_format_event_datetime` so missing components come from metadata.

**`_dateparser_candidates` (`ner_algorithm.py:230-269`)**

- Uses `search_dates` to locate date/time-like spans in the text, then `DateDataParser` to interpret those spans.
- Parser settings use Manila timezone, timezone-aware values, past preference, and time-period detection. When metadata is available, its local wall time becomes `RELATIVE_BASE`; this is how relative terms are anchored to the post's date.
- Uses strict parsing for date-only results to reduce accidental parsing of ordinary words as dates.
- If the search library split a clock around punctuation, tries extending the phrase by one adjacent non-space token and keeps the extension if it parses. This is parser-driven and does not test for particular words.
- Marks `period == "time"` as a time-only result. Other periods count as containing a date. `_format_event_datetime` fills the missing component from metadata.
- Sorts candidates by where they occur in the post, preferring a longer phrase at the same starting offset.

**`_translate_temporal_text` (`ner_algorithm.py:311-317`)**

- Uses `detect_source_language_code` and `translate_text` from `local_models.py`; results are cached for repeated text.
- The language is detected dynamically. In the current helper, English maps to `eng_Latn`, Tagalog to `tgl_Latn`, and other detections default to `ceb_Latn` before local NLLB translation to English.
- It contains no manually enumerated date/time words or translated phrase dictionary. Translation can still be imperfect, so translated parses should be understood as a fallback result and checked with the stored `translation` and `translated_phrase` fields.

**`extract_vague_datetime` (`ner_algorithm.py:319-342`)**

- Runs both recognizer and dateparser on the original text first.
- Returns an explicit time/datetime found in the original text before trying translation.
- If no original explicit result exists, translates the text, runs the same parsers on that translation, then prefers any explicit result across original and translated candidates.
- If there is no explicit result, returns the first date-like candidate in the original/translated candidate order. Returns `None` when there is no reference timestamp or nothing parses.

**`extract_explicit_datetime` (`ner_algorithm.py:345-350`)**

- Despite its function name, it is the public wrapper for both explicit and vague date/time extraction.
- Converts a naive metadata timestamp as UTC, converts the reference to Manila time, and calls `extract_vague_datetime`.
- If extraction returns nothing, returns the metadata timestamp with source `fallback`; if the reference is also unavailable, returns an `unresolved` result.

### Date/time source labels and examples

| Text/parser result | Stored source | Resulting event date/time |
|---|---|---|
| Full date and clock parsed | `explicit` | Parsed local date and time |
| Date parsed, no clock | `vague_date` | Parsed date + metadata time |
| Clock parsed, no date | `explicit` | Metadata date + parsed clock |
| No usable expression, metadata exists | `fallback` | Metadata date and time |
| No usable expression, no metadata | `unresolved` | No event date/time |

The source label describes where the date/time came from, not whether every component was typed in the post. A `vague_date` result may have a filled-in event time from metadata.

## 4. Event clustering: exact functions and responsibilities

### Event function flow

`get_selected_results` / `get_table_data` / `get_events` → `_build_event_lookup` → `cluster_posts_into_events` → `build_distance_matrix` → `is_same_event` → `haversine_km` → `build_event_summary` → `resolve_event_datetime` and `pick_event_location`.

**Important separation:** location centroid disambiguation (`resolve_via_spatiotemporal_centroid`) decides which gazetteer candidate best fits nearby high-confidence locations. Event clustering (`cluster_posts_into_events`) groups posts that already have a validated location. They are not the same operation.

### `ner_routes.py` event functions

**`_location_rank` (`ner_routes.py:301-304`)**

- Ranks locations first by confidence (`HIGH`, `MEDIUM`, `LOW`) and then by count of populated breadcrumb levels.
- Used to pick one representative location per post for clustering and event output.

**`_build_event_lookup` (`ner_routes.py:445-529`)**

- Loads posts whose `disaster_type` is in `DISASTER_TYPES`, which have at least one resolved `ner_location` and a valid `created_at`.
- Picks the post's best location with `_location_rank` and creates a compact cluster input with disaster type, coordinates, confidence, author, posted timestamp, and extracted date/time fields.
- Calls `cluster_posts_into_events`, then `build_event_summary` for each cluster.
- Assigns response-local IDs such as `event-0`; builds a post-ID-to-event-ID mapping.
- Returns event coordinates/place/type/date/time, post count, and member post details.
- The extracted event datetime is used for event summary time. Pairwise cluster membership uses the post's `created_at` metadata timestamp.

**`get_selected_results` (`ner_routes.py:548-559`)**

- Builds event data for the selected post IDs and returns the selected post rows plus summaries from the same lookup.

**`get_table_data` (`ner_routes.py:562-584`)**

- Returns paginated post rows and their event IDs.

**`get_events` (`ner_routes.py:587-593`)**

- Returns only the event summaries.

### `ner_event_clustering.py` functions

**Fixed thresholds at module top (`ner_event_clustering.py:5-6`)**

- `MAX_DISTANCE_KM = 2.0`: maximum pairwise separation for two posts to be directly compatible.
- `MAX_TIME_HOURS = 6.0`: maximum difference between their posted metadata timestamps.
- These are fixed algorithm parameters in the current implementation, not learned model outputs. They need to be explained as the chosen spatial/time window and recalibrated if evaluation shows they are too broad or narrow.

**`haversine_km` (`ner_event_clustering.py:9-16`)**

- Computes great-circle distance between two latitude/longitude points using Earth's radius `6371.0` km.

**`is_same_event` (`ner_event_clustering.py:18-34`)**

- Returns `False` when disaster types differ.
- Otherwise computes Haversine distance and rejects pairs over 2 km.
- Then compares absolute difference in `created_at`; rejects pairs over six hours.
- Returns `True` only if all three checks pass: same disaster type, close enough, and close enough in posted time.

**`build_distance_matrix` (`ner_event_clustering.py:36-45`)**

- Creates an `n × n` symmetric matrix. Compatible post pairs get `0.0`, incompatible pairs `1.0`.
- This is a binary compatibility matrix, not a matrix of kilometer or hour values.

**`cluster_posts_into_events` (`ner_event_clustering.py:48-61`)**

- Runs scikit-learn DBSCAN with a precomputed matrix, `eps=0.5`, `min_samples=1`.
- Since compatible pairs have distance zero and incompatible pairs one, DBSCAN connects compatible posts. `min_samples=1` means every eligible input can form at least a one-post cluster.
- DBSCAN connectivity is transitive: if A matches B and B matches C, all can be in one cluster even if A does not directly match C. Thus the 2 km / six-hour rules define pair links, not a guarantee that every pair inside a final cluster satisfies both thresholds.

**`resolve_event_datetime` (`ner_event_clustering.py:63-87`)**

- Finds members with source `explicit` or `vague_date` and a non-null extracted `event_datetime`; uses the earliest of those as the cluster's anchor time.
- If there is such an anchor, members previously marked unresolved are set to `inferred` at that time in the in-memory cluster input.
- If no member has an extracted date/time, uses the earliest posted `created_at`. Unresolved members are filled from their own created-at timestamp and marked fallback.
- This method changes the cluster payload in memory; it does not update those MongoDB documents.

**`pick_event_location` (`ner_event_clustering.py:90-99`)**

- Selects the highest-confidence member location (`HIGH` before `MEDIUM` before `LOW`) and returns its place and coordinates.
- It does not average all cluster coordinates.

**`build_event_summary` (`ner_event_clustering.py:102-112`)**

- Combines the first member's disaster type, chosen location, resolved event datetime, member posts, and distinct related author handles.
- Assumes all members share the same disaster type because `is_same_event` enforces it.

### What event clustering does not use

- It does not use post-text date/time to decide whether two posts match; it uses `created_at` for the six-hour pairwise condition.
- It does not group posts solely because the disaster type is the same.
- It does not calculate a central/average event location; it picks one member location by confidence.
- Posts without any resolved location or valid posted timestamp are excluded by `_build_event_lookup` and cannot enter an event cluster.

## 5. Dashboard connection

**`spatiotemporal.js:22-52` — `loadSpatiotemporalData`**

- Sends selected post IDs to `/process-all?limit=100` repeatedly until no posts remain.
- Then requests `/results` and returns rows and events.
- The API address is configured as `http://127.0.0.1:8000/ner` in this file.

**`spatiotemporal.js:3-19` — `locationDisplayItems`**

- Formats resolved locations from their breadcrumb and matched name.
- Adds ambiguous candidates as unresolved display items, so uncertainty is visible rather than silently dropped.

**`NERTableView.jsx:19-34` — `getStrictDateTime` / `getEventDateTime`**

- Uses stored event date/time first; falls back to normalized datetime and then posted `created_at` for table display.

**`NERMapView.jsx`**

- Renders returned event coordinates and groups/filters those summaries for map display. It does not rerun geographic resolution or event clustering in the browser.

## 6. Field meanings for explaining a result

| Field | Meaning |
|---|---|
| `ner_locations` | Resolved gazetteer locations, with coordinates, confidence/source, PSGC, breadcrumb, and method |
| `ner_ambiguous_locations` | Gazetteer candidates that remain unresolved |
| `ner_datetime` | Provenance details: extracted phrase/resolution or metadata fallback explanation |
| `ner_datetime_source` | `explicit`, `vague_date`, `fallback`, `unresolved`, or later in-memory event status such as `inferred` |
| `ner_event_datetime` | Datetime value used to describe the event timing |
| `ner_event_date` / `ner_event_time` | Manila-local date and time strings used by the table and event summary |
| `event_id` | Response-generated grouping ID for a clustered event; not a stable database event identifier |

## 7. Short presentation explanation

“For each post, the backend uses spaCy to propose location spans and validates those spans against a Cebu PSGC gazetteer. If the post has no location, it checks the author bio and then up to ten earlier authored posts. Ambiguous place names can be narrowed using a parent municipality or nearby high-confidence anchors. For time, it parses the post timestamp as the Manila-local reference, tries date/time recognizers on the text, optionally translates non-English text with the local model and parses again, and fills any missing date or time component from post metadata. If no text expression parses, it uses the metadata timestamp. Finally, eligible disaster posts with resolved coordinates are linked when they have the same disaster type and are within the configured 2 km and six-hour pairwise thresholds. DBSCAN groups those links; the event summary chooses a representative location and event time.”
