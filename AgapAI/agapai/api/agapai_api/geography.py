import json
import re
from pathlib import Path

PSGC_GAZETTEER_PATH = Path(__file__).resolve().parent / "cebu_gazetteer_final.json"

_DATELINE_PATTERN = re.compile(r'^[A-Z][A-Z.\s]{1,40}(?:,\s*[A-Za-z][A-Za-z.\s]{1,30})?\s*[-–—]+\s+')


def strip_dateline(text):
    if not text:
        return text
    return _DATELINE_PATTERN.sub('', text, count=1)


def load_psgc_cebu_gazetteer(gazetteer_path=None):

    path = Path(gazetteer_path) if gazetteer_path else PSGC_GAZETTEER_PATH
    with open(path, "r", encoding="utf-8") as gazetteer_file:
        return json.load(gazetteer_file)

DEFAULT_CEBU_EXPLICIT_MARKERS = (
    "cebu", "cebu city", "cebu province", "province of cebu", "sugbo", "probinsya sa sugbo",
    "metro cebu", "greater cebu",
)


def _normalize_name_variants(names):
    variants = set()
    for name in names:
        cleaned = re.sub(r'\s+', ' ', str(name).strip().lower())
        if not cleaned:
            continue
        variants.add(cleaned)
        if "-" in cleaned:
            variants.add(cleaned.replace("-", " "))
    return variants


def extract_psgc_cebu_registry(gazetteer=None):

    gazetteer = gazetteer if gazetteer is not None else load_psgc_cebu_gazetteer()

    cities_municipalities_raw = _normalize_name_variants(gazetteer.get("cities_municipalities", []))
    barangays_raw = _normalize_name_variants(gazetteer.get("barangays", {}).keys())
    out_of_bounds = _normalize_name_variants(gazetteer.get("out_of_bounds_blacklist", []))
    explicit_context_markers = _normalize_name_variants(
        gazetteer.get("explicit_context_markers") or DEFAULT_CEBU_EXPLICIT_MARKERS
    )

    cities_municipalities = cities_municipalities_raw - out_of_bounds
    barangays = barangays_raw - out_of_bounds
    cebu_locations = (cities_municipalities_raw | barangays_raw) - out_of_bounds

    return {
        "cities_municipalities": cities_municipalities,
        "barangays": barangays,
        "all_locations": cebu_locations,
        "out_of_bounds": out_of_bounds,
        "explicit_context_markers": explicit_context_markers,
        "philippine_context_markers": _normalize_name_variants(gazetteer.get("philippine_context_markers", [])),
        "landmarks": _normalize_name_variants(gazetteer.get("landmarks", [])),
        "ambiguous_homonym_localities": _normalize_name_variants(gazetteer.get("ambiguous_homonym_localities", [])),
        "canonical_display_names": _build_canonical_display_names(cebu_locations),
    }


def _build_canonical_display_names(location_names):

    canonical_map = {}
    for name in location_names:
        if "-" not in name:
            continue
        canonical = "-".join(segment.title() for segment in name.split("-"))
        canonical_map[name] = canonical
        canonical_map[name.replace("-", " ")] = canonical
    return canonical_map


_PSGC_REGISTRY = extract_psgc_cebu_registry()

PSGC_CEBU_CITIES_MUNICIPALITIES = _PSGC_REGISTRY["cities_municipalities"]
PSGC_CEBU_BARANGAYS = _PSGC_REGISTRY["barangays"]
PSGC_CEBU_ALL_LOCATIONS = _PSGC_REGISTRY["all_locations"]
PSGC_OUT_OF_BOUNDS_BLACKLIST = _PSGC_REGISTRY["out_of_bounds"]
PSGC_CANONICAL_DISPLAY_NAMES = _PSGC_REGISTRY["canonical_display_names"]

CEBU_LOCALITY_MARKERS = PSGC_CEBU_CITIES_MUNICIPALITIES

CEBU_EXPLICIT_MARKERS = _PSGC_REGISTRY["explicit_context_markers"]
PHILIPPINE_CONTEXT_MARKERS = _PSGC_REGISTRY["philippine_context_markers"]
CEBU_LANDMARK_MARKERS = _PSGC_REGISTRY["landmarks"]
AMBIGUOUS_CEBU_LOCALITIES = _PSGC_REGISTRY["ambiguous_homonym_localities"]

CEBU_CONTEXT_MARKERS = CEBU_EXPLICIT_MARKERS | PSGC_CEBU_ALL_LOCATIONS | CEBU_LANDMARK_MARKERS
CEBU_SECONDARY_MARKERS = CEBU_EXPLICIT_MARKERS | CEBU_LANDMARK_MARKERS | PHILIPPINE_CONTEXT_MARKERS


def load_cebu_geographic_registry():
    contextual_administrative_markers = {
        "brgy", "barangay", "sitio", "purok", "kalye", "street", "st", "bayan",
        "bldg", "provincial", "poblacion",
    }

    registry = CEBU_EXPLICIT_MARKERS | contextual_administrative_markers | PSGC_CEBU_ALL_LOCATIONS | CEBU_LANDMARK_MARKERS
    print(f"SUCCESS: Indexed {len(registry)} Cebu Province location markers only.")
    return registry


CEBU_GEOGRAPHIC_REGISTRY = load_cebu_geographic_registry()


def contains_phrase(text, phrase):
    escaped = re.escape(phrase).replace(r'\ ', r'[\s.-]+')
    return re.search(r'\b' + escaped + r'\b', text, flags=re.IGNORECASE) is not None


def find_matching_phrases(text, phrases):
    return sorted({phrase.title() for phrase in phrases if contains_phrase(text, phrase)})


def format_location_name(phrase):
    lowered = re.sub(r'\s+', ' ', phrase.strip().lower())
    if lowered in PSGC_CANONICAL_DISPLAY_NAMES:
        return PSGC_CANONICAL_DISPLAY_NAMES[lowered]
    return phrase.title()


def find_ordered_matching_phrases(text, phrases):
    matches = []
    for phrase in phrases:
        escaped = re.escape(phrase).replace(r'\ ', r'[\s.-]+')
        match = re.search(r'\b' + escaped + r'\b', text, flags=re.IGNORECASE)
        if match:
            matches.append((match.start(), -len(phrase), format_location_name(phrase)))
    return [match_text for _start, _length, match_text in sorted(matches)]


def format_context_location(value):
    cleaned_value = re.sub(r'\s+', ' ', value.strip())
    cleaned_value = re.sub(r'\s+(cebu city|cebu)$', '', cleaned_value, flags=re.IGNORECASE)
    cleaned_value = re.sub(r'^(brgy)\.?\s*[,.-]?\s*', 'Brgy. ', cleaned_value, flags=re.IGNORECASE)
    cleaned_value = re.sub(r'^(barangay)\s*[,.-]?\s*', 'Barangay ', cleaned_value, flags=re.IGNORECASE)
    return cleaned_value.title().replace("Brgy.", "Brgy.")


def has_cebu_context(text):

    if not text:
        return False, []

    explicit_matches = find_matching_phrases(text, CEBU_EXPLICIT_MARKERS)
    landmark_matches = find_matching_phrases(text, CEBU_LANDMARK_MARKERS)
    locality_matches = find_matching_phrases(text, PSGC_CEBU_ALL_LOCATIONS)
    secondary_matches = find_matching_phrases(text, CEBU_SECONDARY_MARKERS)
    ambiguous_matches = find_matching_phrases(text, AMBIGUOUS_CEBU_LOCALITIES)

    if explicit_matches or landmark_matches:
        return True, sorted(set(explicit_matches + landmark_matches + locality_matches))

    unambiguous_locality_matches = [
        match for match in locality_matches if match.lower() not in AMBIGUOUS_CEBU_LOCALITIES
    ]
    if unambiguous_locality_matches:
        return True, locality_matches

    if locality_matches and secondary_matches and not ambiguous_matches:
        return True, sorted(set(locality_matches + secondary_matches))

    if locality_matches and len(locality_matches) >= 2:
        return True, locality_matches

    return False, locality_matches


def find_out_of_bounds_matches(text):

    if not text:
        return []
    return find_matching_phrases(text, PSGC_OUT_OF_BOUNDS_BLACKLIST)


def is_out_of_bounds(text):
    """True if text names any place in the out_of_bounds_blacklist."""
    return bool(find_out_of_bounds_matches(text))


def extract_location_name(text):

    text = strip_dateline(text)

    context_pattern = r'\b(purok|brgy|barangay|sitio|kalye|street|st|bayan|poblacion)\b\.?\s*[,.-]?\s*(?:\d+\s*)?[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,2}'
    match = re.search(context_pattern, text, flags=re.IGNORECASE)
    if match:
        context_location = format_context_location(match.group(0))
        locality_matches = find_ordered_matching_phrases(text, PSGC_CEBU_ALL_LOCATIONS - {"cebu city"})
        if "Poblacion" in context_location and locality_matches:
            return locality_matches[0]
        return context_location

    specific_localities = PSGC_CEBU_ALL_LOCATIONS - {"cebu city"}
    locality_matches = find_ordered_matching_phrases(text, specific_localities)
    if locality_matches:
        return locality_matches[0]

    landmark_matches = find_ordered_matching_phrases(text, CEBU_LANDMARK_MARKERS)
    if landmark_matches:
        return landmark_matches[0]

    phrase_matches = find_ordered_matching_phrases(text, CEBU_GEOGRAPHIC_REGISTRY)
    if phrase_matches:
        return phrase_matches[0]

    words = re.findall(r'\b\w+\b', text.lower())
    for word in words:
        if word in CEBU_GEOGRAPHIC_REGISTRY:
            return word.capitalize()

    return None

CEBU_MUNICIPALITIES_CITIES = PSGC_CEBU_CITIES_MUNICIPALITIES


def get_cebu_location_search_terms(include_barangays=False):

    location_names = PSGC_CEBU_CITIES_MUNICIPALITIES
    if include_barangays:
        location_names = location_names | PSGC_CEBU_BARANGAYS

    anchored_terms = {f"{format_location_name(name)} Cebu" for name in location_names}

    raw_gazetteer = load_psgc_cebu_gazetteer()
    capital_present = any(
        str(entry).strip().lower() == "cebu"
        for entry in raw_gazetteer.get("cities_municipalities", [])
    )
    if capital_present:
        anchored_terms.add("Cebu City")

    landmark_terms = {
        name.title() if "cebu" in name.lower() else f"{name.title()} Cebu"
        for name in CEBU_LANDMARK_MARKERS
    }

    search_terms = sorted(anchored_terms | landmark_terms)

    print(f"[AGAPAI LOG] Dynamically extracted {len(search_terms)} Cebu location entities from PSGC.")
    return search_terms
