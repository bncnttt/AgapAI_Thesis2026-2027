"""PSGC validation and hierarchy. This GeoJSON is the only location source."""
import json
import re
from functools import lru_cache
from pathlib import Path

from spacy.lang.en.stop_words import STOP_WORDS

GAZETTEER_PATH = Path(__file__).with_name("Cebu_Province_PSGC_Gazetteer.geojson")
MUNICIPAL_LEVEL_TYPES = ("City", "Municipality", "Highly Urbanized City")


def _normalize(name):
    return " ".join(name.casefold().replace("-", " ").split())


@lru_cache(maxsize=1)
def _load_gazetteer():
    with GAZETTEER_PATH.open(encoding="utf-8") as source:
        features = json.load(source)["features"]
    lookup = {}
    administrative_words = {word.casefold() for kind in MUNICIPAL_LEVEL_TYPES for word in kind.split()}
    for feature in features:
        properties = feature["properties"]
        lon, lat = feature["geometry"]["coordinates"]
        entry = {**properties, "latitude": lat, "longitude": lon}
        name = _normalize(entry["name"])
        variants = {name}
        if entry["type"] in MUNICIPAL_LEVEL_TYPES:
            # Derive settlement aliases from official names and type metadata.
            # Function words come from spaCy; no place-name list is maintained.
            base = " ".join(word for word in name.split() if word not in administrative_words | STOP_WORDS)
            if base:
                variants.update((base, f"{base} {entry['type'].split()[-1].casefold()}"))
        for variant in variants:
            lookup.setdefault(variant, []).append(entry)
    return lookup


def find_exact(name):
    key = _normalize(name)
    matches = _load_gazetteer().get(key, [])
    # An official name outranks an alias of another administrative area.
    official = [entry for entry in matches if _normalize(entry["name"]) == key]
    return official if any(entry["type"] != "Barangay" for entry in official) else matches


def names_in_entity(entity_text):
    """Validate complete word sequences inside a model-predicted entity only."""
    stripped = entity_text.strip()
    if find_exact(stripped):
        start = entity_text.index(stripped)
        return [(start, start + len(stripped), stripped)]
    tokens = list(re.finditer(r"\w+(?:[-']\w+)*", entity_text))
    names = []
    start = 0
    while start < len(tokens):
        for end in range(len(tokens), start, -1):
            phrase = entity_text[tokens[start].start():tokens[end - 1].end()]
            if find_exact(phrase):
                names.append((tokens[start].start(), tokens[end - 1].end(), phrase))
                start = end
                break
        else:
            start += 1
    return names


def find_by_parent(name, parent_hint):
    matches = find_exact(name)
    if not parent_hint:
        return matches
    parent_names = {entry["name"] for entry in find_exact(parent_hint) if entry["type"] in MUNICIPAL_LEVEL_TYPES}
    narrowed = [entry for entry in matches if entry.get("parent_city_municipality") in parent_names]
    return narrowed or matches


def build_breadcrumb(entry):
    province = next(
        item["name"] for entries in _load_gazetteer().values() for item in entries
        if item["type"] == "Province"
    )
    # Region is absent from this file; preserve the four-level display schema.
    if entry["type"] == "Province":
        return [None, province, None, None]
    if entry["type"] in MUNICIPAL_LEVEL_TYPES:
        return [None, province, entry["name"], None]
    return [None, province, entry.get("parent_city_municipality"), entry["name"]]
