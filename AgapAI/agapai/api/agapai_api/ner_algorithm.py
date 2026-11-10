"""Model candidates -> PSGC validation -> source confidence -> library time parsing.

Sections follow that order. ner_routes.py supplies posts and author context;
ner_event_clustering.py groups the resolved results into events.
"""
import re
from datetime import datetime, time, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import spacy
from dateparser import DateDataParser
from dateparser.search import search_dates
from huggingface_hub import snapshot_download
from recognizers_date_time import recognize_datetime, Culture

from agapai_api.ner_gazetteer_loader import (
    MUNICIPAL_LEVEL_TYPES, build_breadcrumb, find_by_parent, find_exact, names_in_entity,
)
from agapai_api.ner_event_clustering import haversine_km

NER_ALGORITHM_VERSION = "2026-10-29.5"
SPATIOTEMPORAL_MAX_DISTANCE_KM = 5.0
AUTHOR_HISTORY_LIMIT = 10
LOCAL_TIMEZONE = ZoneInfo("Asia/Manila")
_nlp = None


# 1. Model predictions. Reuse one Doc for location and raw-entity extraction.
def get_nlp():
    global _nlp
    if _nlp is None:
        model_path = snapshot_download(
            repo_id="ljvmiranda921/tl_calamancy_md", revision="0.2.0", ignore_patterns=["*.whl"],
        )
        _nlp = spacy.load(model_path)
    return _nlp


def extract_location_entities(text, doc=None):
    if not text or not text.strip():
        return []
    doc = get_nlp()(text) if doc is None else doc
    candidates = [ent for ent in doc.ents if ent.label_ in ("LOC", "ORG")]
    # Informal addresses are often lowercase and can be mistagged as verbs.
    # Use the model's oblique/modifier/list structure, independent of casing.
    address_relations = {"obl", "nmod", "compound", "flat", "appos", "conj", "list"}
    start = None
    for index, token in enumerate(doc):
        if token.pos_ == "PROPN" or (
            token.pos_ in ("NOUN", "VERB") and token.dep_.split(":")[0] in address_relations
        ):
            if start is None:
                start = index
        elif start is not None:
            candidates.append(doc[start:index])
            start = None
    if start is not None:
        candidates.append(doc[start:])
    mentions = {}
    for candidate in candidates:
        for start, end, name in names_in_entity(candidate.text):
            offset = candidate.start_char
            mentions[(offset + start, offset + end)] = name
    ordered = sorted(mentions)
    ordered = [(start, end) for start, end in ordered if not any(
        outer_start <= start and end <= outer_end and (start, end) != (outer_start, outer_end)
        for outer_start, outer_end in ordered
    )]
    entities, consumed = [], set()
    for index, span in enumerate(ordered):
        if index in consumed:
            continue
        name = mentions[span]
        children = find_exact(name)
        previous = span
        for following_index in range(index + 1, len(ordered)):
            following = ordered[following_index]
            if not re.fullmatch(r"[\s,]+", text[previous[1]:following[0]]):
                break
            parents = find_exact(mentions[following])
            parent_names = {parent["name"] for parent in parents if parent["type"] in MUNICIPAL_LEVEL_TYPES}
            if any(child.get("parent_city_municipality") in parent_names for child in children):
                name += ", " + mentions[following]
            elif not (any(parent["type"] == "Province" for parent in parents)
                      and all(child["type"] != "Province" for child in children)):
                break
            consumed.add(following_index)
            previous, children = following, parents
        entities.append(name)
    return list(dict.fromkeys(entities))


def extract_raw_entities(text, doc=None):
    if not text or not text.strip():
        return {"locations": [], "persons": [], "organizations": []}
    doc = get_nlp()(text) if doc is None else doc
    return {
        key: list(dict.fromkeys(ent.text.strip() for ent in doc.ents if ent.label_ == label))
        for key, label in (("locations", "LOC"), ("persons", "PER"), ("organizations", "ORG"))
    }


# 2. Gazetteer validation and same-post administrative hierarchy.
def resolve_location_text(location_text, municipality_hint=None):
    fragments = [name for _, _, name in names_in_entity(location_text)]
    if not fragments:
        return None
    hint = fragments[1] if len(fragments) > 1 else municipality_hint
    matches = find_by_parent(fragments[0], hint)
    if len(matches) > 1:
        settlements = [entry for entry in matches if entry["type"] in MUNICIPAL_LEVEL_TYPES]
        if len(settlements) == 1:
            matches = settlements
    if len(matches) == 1:
        entry = matches[0]
        return {**entry, "matched_name": entry["name"], "breadcrumb": build_breadcrumb(entry)}
    return {
        "ambiguous": True, "candidate_count": len(matches),
        "candidates": [{**entry, "breadcrumb": build_breadcrumb(entry)} for entry in matches],
    }


def _location(entry, text, confidence="HIGH", source="post_text", method="gazetteer"):
    return {
        "location_text": text, "matched_name": entry.get("matched_name", entry["name"]),
        "coordinates": {"lat": entry["latitude"], "lon": entry["longitude"]},
        "psgc_code": entry["psgc_code"], "breadcrumb": entry["breadcrumb"],
        "confidence": confidence, "confidence_source": source, "resolution_method": method,
    }


def _location_result(locations, ambiguities):
    unique = {loc["psgc_code"]: loc for loc in locations}
    unresolved = {
        tuple(sorted(candidate["psgc_code"] for candidate in item["candidates"])): item
        for item in ambiguities if item["candidates"]
    }
    return {
        "status": "RESOLVED" if unique else "AMBIGUOUS_REVIEW_QUEUE" if unresolved else "LOW_CONFIDENCE_REVIEW_QUEUE",
        "locations": list(unique.values()), "ambiguous_locations": list(unresolved.values()),
    }


def extract_and_resolve(text, doc=None):
    first_pass = [(name, resolve_location_text(name)) for name in extract_location_entities(text, doc)]
    hints = [result["matched_name"] for _, result in first_pass
             if result and not result.get("ambiguous") and result["type"] in MUNICIPAL_LEVEL_TYPES]
    locations, ambiguities = [], []
    for name, result in first_pass:
        if not result:
            continue
        if result.get("ambiguous"):
            alternatives = {}
            for hint in hints:
                candidate = resolve_location_text(name, hint)
                if candidate and not candidate.get("ambiguous"):
                    alternatives[candidate["psgc_code"]] = candidate
            if len(alternatives) == 1:
                result = next(iter(alternatives.values()))
        if result.get("ambiguous"):
            ambiguities.append({"location_text": name, **result})
        else:
            locations.append(_location(result, name))
    result = _location_result(locations, ambiguities)
    return result["locations"], result["ambiguous_locations"]


# 3. Source cascade: post (HIGH), bio (MEDIUM), previous ten posts (LOW).
def resolve_via_spatiotemporal_centroid(candidates, nearby_high_confidence_points):
    if not candidates or not nearby_high_confidence_points:
        return None
    lat = sum(point["lat"] for point in nearby_high_confidence_points) / len(nearby_high_confidence_points)
    lon = sum(point["lon"] for point in nearby_high_confidence_points) / len(nearby_high_confidence_points)
    nearest = min(candidates, key=lambda entry: haversine_km(entry["latitude"], entry["longitude"], lat, lon))
    return nearest if haversine_km(nearest["latitude"], nearest["longitude"], lat, lon) <= SPATIOTEMPORAL_MAX_DISTANCE_KM else None


def resolve_post_location(post_text, author_bio=None, previous_post_texts=None,
                          nearby_high_confidence_points=None, doc=None):
    locations, ambiguous = extract_and_resolve(post_text, doc)
    if locations or ambiguous:
        # A recognized post location ends the author-source cascade, even if
        # its parent is ambiguous. Never substitute a bio/history location.
        remaining = []
        for item in ambiguous:
            nearest = resolve_via_spatiotemporal_centroid(item["candidates"], nearby_high_confidence_points)
            if nearest:
                locations.append(_location(nearest, item["location_text"], method="spatiotemporal_centroid"))
            else:
                remaining.append(item)
        return _location_result(locations, remaining)

    sources = [(author_bio, "MEDIUM", "author_bio")]
    sources.extend((text, "LOW", "author_previous_post") for text in (previous_post_texts or [])[:AUTHOR_HISTORY_LIMIT])
    for text, confidence, source in sources:
        found, uncertain = extract_and_resolve(text)
        if found or uncertain:
            for location in found:
                location.update(confidence=confidence, confidence_source=source)
            return _location_result(found, uncertain)
    return _location_result([], [])


# 4. Library time extraction; no custom temporal vocabulary or phrase rules.
def _format_event_datetime(parsed_datetime, source, reference_date=None, has_date=True, has_time=True):
    if parsed_datetime is not None:
        if parsed_datetime.tzinfo is None:
            parsed_datetime = parsed_datetime.replace(tzinfo=LOCAL_TIMEZONE)
        else:
            parsed_datetime = parsed_datetime.astimezone(LOCAL_TIMEZONE)

        if reference_date is not None:
            reference_date = reference_date.astimezone(LOCAL_TIMEZONE)
            event_date = parsed_datetime.date() if has_date else reference_date.date()
            event_time = (
                parsed_datetime.timetz().replace(tzinfo=None)
                if has_time else reference_date.timetz().replace(tzinfo=None)
            )
            parsed_datetime = datetime.combine(event_date, event_time).replace(tzinfo=LOCAL_TIMEZONE)

    return {
        "event_datetime": parsed_datetime,
        "event_date": parsed_datetime.date().isoformat() if parsed_datetime else None,
        "event_time": parsed_datetime.strftime("%I:%M:%S %p") if parsed_datetime else None,
        "datetime_source": source, "source": source,
    }


def _dateparser_candidates(post_text, reference_date=None):
    if not post_text or not post_text.strip():
        return []
    settings = {
        "TIMEZONE": str(LOCAL_TIMEZONE), "RETURN_AS_TIMEZONE_AWARE": True,
        "RETURN_TIME_AS_PERIOD": True, "PREFER_DATES_FROM": "past",
    }
    if reference_date is not None:
        # dateparser applies TIMEZONE to this local wall-clock reference. A
        # ZoneInfo-aware reference breaks time-only parsing in dateparser 1.4.
        settings["RELATIVE_BASE"] = reference_date.replace(tzinfo=None)
    results = []
    parser = DateDataParser(settings=settings)
    strict_parser = DateDataParser(settings={**settings, "STRICT_PARSING": True})
    for phrase, _ in search_dates(post_text, settings=settings) or []:
        parsed = parser.get_date_data(phrase)
        offset = post_text.find(phrase)
        # The library's search can split punctuation inside a clock. Test
        # the adjacent token with the parser without identifying any words.
        if offset >= 0:
            suffix = re.search(r"\S+", post_text[offset + len(phrase):])
            if suffix:
                extended = post_text[offset:offset + len(phrase) + suffix.end()]
                candidate = parser.get_date_data(extended)
                if candidate.date_obj is not None:
                    parsed, phrase = candidate, extended
        if parsed.date_obj is None:
            continue
        has_time = parsed.period == "time"
        has_date = parsed.period != "time"
        if not has_time and strict_parser.get_date_data(phrase).date_obj is None:
            continue
        result = _format_event_datetime(
            parsed.date_obj, "explicit" if has_time else "vague_date",
            reference_date, has_date=has_date, has_time=has_time,
        )
        result["raw_phrase"] = phrase
        results.append((offset, -len(phrase), result))
    return [item[2] for item in sorted(results, key=lambda item: item[:2])]


def _recognized_datetime_candidates(text, reference_date):
    """The library supplies expressions, types, ranges and normalized values."""
    results = []
    for match in recognize_datetime(text, Culture.English, reference=reference_date.replace(tzinfo=None)):
        values = match.resolution.get("values", []) if match.resolution else []
        unique = {(value.get("type"), value.get("value"), value.get("start"), value.get("end")): value for value in values}
        if len(unique) != 1:
            continue  # an unspecified AM/PM must not become a guessed clock
        value = next(iter(unique.values()))
        kind = value.get("type")
        normalized = value.get("value")
        has_date = kind != "time"
        has_time = kind in ("time", "datetime")
        if kind == "time":
            # Reject malformed clock syntax using the date library rather than
            # accepting a recognizer's repair of an invalid clock.
            clock = DateDataParser().get_date_data(match.text)
            if clock.date_obj is None:
                continue
            parsed = datetime.combine(reference_date.date(), time.fromisoformat(normalized))
        elif kind in ("date", "datetime"):
            parsed = datetime.fromisoformat(normalized)
        elif kind in ("daterange", "datetimerange") and value.get("start") and value.get("end"):
            start, end = datetime.fromisoformat(value["start"]), datetime.fromisoformat(value["end"])
            if start.date() != end.date():
                continue
            parsed = datetime.combine(start.date(), time.min)
        else:
            continue
        result = _format_event_datetime(
            parsed, "explicit" if has_time else "vague_date",
            reference_date, has_date=has_date, has_time=has_time,
        )
        result["raw_phrase"] = match.text
        result["temporal_resolution"] = value
        results.append(result)
    return results


@lru_cache(maxsize=256)
def _translate_temporal_text(text):
    # Reuse the project's local translation model. No external translation API,
    # hand-written translations, place matching, or word lists are involved.
    from agapai_api.local_models import detect_source_language_code, translate_text
    language = detect_source_language_code(text)
    return translate_text(text, language) if language != "eng_Latn" else None


def extract_vague_datetime(post_text, reference_date=None):
    if not post_text or not post_text.strip() or reference_date is None:
        return None
    original = (
        _recognized_datetime_candidates(post_text, reference_date)
        + _dateparser_candidates(post_text, reference_date)
    )
    explicit = next((result for result in original if result["source"] == "explicit"), None)
    if explicit:
        return explicit
    translated = _translate_temporal_text(post_text)
    translated_results = []
    if translated and translated != post_text:
        translated_results = (
            _recognized_datetime_candidates(translated, reference_date)
            + _dateparser_candidates(translated, reference_date)
        )
        for result in translated_results:
            result["translated_phrase"] = result["raw_phrase"]
            result["raw_phrase"] = post_text
            result["translation"] = translated
    # Explicit clocks/datetimes outrank vague mentions elsewhere in the post.
    results = original + translated_results
    return next((result for result in results if result["source"] == "explicit"), results[0] if results else None)


def extract_explicit_datetime(post_text, reference_date=None):
    if reference_date is not None:
        if reference_date.tzinfo is None:
            reference_date = reference_date.replace(tzinfo=timezone.utc)
        reference_date = reference_date.astimezone(LOCAL_TIMEZONE)
    extracted = extract_vague_datetime(post_text, reference_date)
    if extracted:
        return extracted
    return _format_event_datetime(reference_date, "fallback" if reference_date else "unresolved")
