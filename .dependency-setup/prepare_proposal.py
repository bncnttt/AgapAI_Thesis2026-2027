from pathlib import Path
import difflib

root = Path("C:/AgapAI/AgapAI")
requirements = "agapai/api/requirements.txt"
algorithm = "agapai/api/agapai_api/ner_algorithm.py"
original_requirements = (root / requirements).read_text(encoding="utf-8")
proposed_requirements = original_requirements.replace(
    "calamancy\n", "spacy>=3.8.3\nhuggingface-hub>=0.20.0\n"
) + "\nnetworkx\n"
original_algorithm = (root / algorithm).read_text(encoding="utf-8")
proposed_algorithm = original_algorithm.replace(
    "import calamancy\n", "import spacy\nfrom huggingface_hub import snapshot_download\n"
).replace(
    '        _nlp = calamancy.load("tl_calamancy_md-0.2.0")',
    '''        model_path = snapshot_download(
            repo_id="ljvmiranda921/tl_calamancy_md",
            revision="0.2.0",
            ignore_patterns=["*.whl"],
        )
        _nlp = spacy.load(model_path)''',
)
patch = "".join(
    "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                               fromfile="a/" + path, tofile="b/" + path))
    for path, old, new in (
        (requirements, original_requirements, proposed_requirements),
        (algorithm, original_algorithm, proposed_algorithm),
    )
)
Path("C:/AgapAI/.dependency-setup/proposed-dependency-fix.patch").write_text(patch, encoding="utf-8")
print("Prepared proposed-dependency-fix.patch; application files are unchanged.")

if __name__ == "__main__":
    import os
    import sys
    os.environ["HF_HUB_OFFLINE"] = "1"
    sys.path.insert(0, str(root / "agapai/api"))
    namespace = {}
    exec(compile(proposed_algorithm.lstrip("\ufeff"), algorithm, "exec"), namespace)
    doc = namespace["get_nlp"]()("May baha sa Talisay, Cebu.")
    assert any(entity.label_ == "LOC" for entity in doc.ents)
    print("Proposed dependency fix: same cached NER model loads and extracts locations.")
