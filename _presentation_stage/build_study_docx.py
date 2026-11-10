from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).parent
SOURCE = ROOT / "AgapAI_Function_by_Function_Study_Notes.md"
OUTPUT = ROOT / "AgapAI_Spatiotemporal_Function_by_Function_Study_Notes.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
DC = "http://purl.org/dc/elements/1.1/"
DCTERMS = "http://purl.org/dc/terms/"
XSI = "http://www.w3.org/2001/XMLSchema-instance"
ET.register_namespace("w", W)
ET.register_namespace("r", R)


def q(local: str) -> str:
    return f"{{{W}}}{local}"


def add_run(parent, text, *, bold=False, code=False, size=None, color=None):
    run = ET.SubElement(parent, q("r"))
    rpr = ET.SubElement(run, q("rPr"))
    if bold:
        ET.SubElement(rpr, q("b"))
    if code:
        fonts = ET.SubElement(rpr, q("rFonts"))
        fonts.set(q("ascii"), "Consolas")
        fonts.set(q("hAnsi"), "Consolas")
    if size:
        sz = ET.SubElement(rpr, q("sz"))
        sz.set(q("val"), str(size))
    if color:
        c = ET.SubElement(rpr, q("color"))
        c.set(q("val"), color)
    chunks = text.split("\n")
    for index, chunk in enumerate(chunks):
        if index:
            ET.SubElement(run, q("br"))
        if chunk:
            t = ET.SubElement(run, q("t"))
            if chunk.startswith(" ") or chunk.endswith(" "):
                t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
            t.text = chunk
    return run


INLINE = re.compile(r"(\*\*.*?\*\*|`[^`]+`)")


def fill_inline(p, text, *, table=False):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            add_run(p, part[2:-2], bold=True, size=18 if table else None)
        elif part.startswith("`") and part.endswith("`"):
            add_run(p, part[1:-1], code=True, size=17 if table else None)
        else:
            add_run(p, part, size=18 if table else None)


def paragraph(parent, text, style=None, *, page_break=False):
    p = ET.SubElement(parent, q("p"))
    ppr = ET.SubElement(p, q("pPr"))
    if style:
        pstyle = ET.SubElement(ppr, q("pStyle"))
        pstyle.set(q("val"), style)
    if page_break:
        ET.SubElement(ppr, q("pageBreakBefore"))
    fill_inline(p, text)
    return p


def cell_text(cell, text, *, header=False):
    p = ET.SubElement(cell, q("p"))
    ppr = ET.SubElement(p, q("pPr"))
    ET.SubElement(ppr, q("spacing")).set(q("after"), "40")
    fill_inline(p, text, table=True)
    if header:
        for run in p.findall(q("r")):
            rpr = run.find(q("rPr"))
            if rpr is None:
                rpr = ET.SubElement(run, q("rPr"))
            if rpr.find(q("b")) is None:
                ET.SubElement(rpr, q("b"))
            color = rpr.find(q("color"))
            if color is None:
                color = ET.SubElement(rpr, q("color"))
            color.set(q("val"), "17365D")


def add_table(body, rows):
    table = ET.SubElement(body, q("tbl"))
    tblpr = ET.SubElement(table, q("tblPr"))
    ET.SubElement(tblpr, q("tblW"), {q("w"): "0", q("type"): "auto"})
    borders = ET.SubElement(tblpr, q("tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        ET.SubElement(borders, q(edge), {q("val"): "single", q("sz"): "4", q("space"): "0", q("color"): "C7D5E5"})
    grid = ET.SubElement(table, q("tblGrid"))
    cols = max(len(row) for row in rows)
    for _ in range(cols):
        ET.SubElement(grid, q("gridCol"), {q("w"): str(9360 // cols)})
    for row_index, row in enumerate(rows):
        tr = ET.SubElement(table, q("tr"))
        if row_index == 0:
            trpr = ET.SubElement(tr, q("trPr"))
            ET.SubElement(trpr, q("tblHeader"))
        for col_index in range(cols):
            tc = ET.SubElement(tr, q("tc"))
            tcpr = ET.SubElement(tc, q("tcPr"))
            ET.SubElement(tcpr, q("tcW"), {q("w"): str(9360 // cols), q("type"): "dxa"})
            if row_index == 0:
                ET.SubElement(tcpr, q("shd"), {q("fill"): "DCE6F1", q("val"): "clear"})
            cell_text(tc, row[col_index] if col_index < len(row) else "", header=(row_index == 0))


def parse_table(lines, start):
    rows = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        raw = lines[index].strip().strip("|")
        cells = [part.strip() for part in raw.split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", part or "-") for part in cells):
            rows.append(cells)
        index += 1
    return rows, index


def styles_xml():
    styles = ET.Element(q("styles"))
    definitions = [
        ("Normal", "Normal", "Calibri", "21", "222222", "140", "120"),
        ("Title", "Title", "Calibri", "36", "17365D", "220", "120"),
        ("Subtitle", "Subtitle", "Calibri", "22", "5B6573", "120", "160"),
        ("Heading1", "Heading 1", "Calibri", "30", "17365D", "300", "140"),
        ("Heading2", "Heading 2", "Calibri", "24", "1F4E79", "220", "100"),
        ("Heading3", "Heading 3", "Calibri", "21", "365F91", "170", "70"),
        ("ListParagraph", "List Paragraph", "Calibri", "21", "222222", "40", "60"),
    ]
    for style_id, name, font, size, color, before, after in definitions:
        style = ET.SubElement(styles, q("style"), {q("type"): "paragraph", q("styleId"): style_id})
        ET.SubElement(style, q("name"), {q("val"): name})
        ppr = ET.SubElement(style, q("pPr"))
        ET.SubElement(ppr, q("spacing"), {q("before"): before, q("after"): after, q("line"): "276", q("lineRule"): "auto"})
        if style_id.startswith("Heading"):
            ET.SubElement(ppr, q("keepNext"))
        if style_id == "Title":
            ET.SubElement(ppr, q("jc"), {q("val"): "center"})
        if style_id == "Subtitle":
            ET.SubElement(ppr, q("jc"), {q("val"): "center"})
        rpr = ET.SubElement(style, q("rPr"))
        ET.SubElement(rpr, q("rFonts"), {q("ascii"): font, q("hAnsi"): font})
        ET.SubElement(rpr, q("sz"), {q("val"): size})
        ET.SubElement(rpr, q("color"), {q("val"): color})
        if style_id != "Normal" and style_id != "Subtitle":
            ET.SubElement(rpr, q("b"))
    return ET.tostring(styles, encoding="utf-8", xml_declaration=True)


def document_xml(markdown):
    doc = ET.Element(q("document"))
    body = ET.SubElement(doc, q("body"))
    lines = markdown.splitlines()
    index = 0
    seen_h1 = False
    while index < len(lines):
        line = lines[index].rstrip()
        if not line.strip():
            index += 1
            continue
        if line.strip().startswith("|"):
            rows, index = parse_table(lines, index)
            if rows:
                add_table(body, rows)
            continue
        heading = re.match(r"^(#{1,3})\s+(.*)$", line)
        if heading:
            depth = len(heading.group(1))
            style = {1: "Heading1", 2: "Heading2", 3: "Heading3"}[depth]
            page_break = depth == 1 and seen_h1
            paragraph(body, heading.group(2), style, page_break=page_break)
            if depth == 1:
                seen_h1 = True
        elif line.startswith("**") and line.endswith("**") and line.count("**") == 2:
            paragraph(body, line.replace("**", ""), "Subtitle")
        elif line.startswith("- "):
            p = ET.SubElement(body, q("p"))
            ppr = ET.SubElement(p, q("pPr"))
            ET.SubElement(ppr, q("pStyle"), {q("val"): "ListParagraph"})
            ET.SubElement(ppr, q("ind"), {q("left"): "360", q("hanging"): "180"})
            add_run(p, "• ")
            fill_inline(p, line[2:])
        else:
            paragraph(body, line)
        index += 1

    sect = ET.SubElement(body, q("sectPr"))
    ET.SubElement(sect, q("pgSz"), {q("w"): "12240", q("h"): "15840"})
    ET.SubElement(sect, q("pgMar"), {q("top"): "900", q("right"): "900", q("bottom"): "900", q("left"): "900", q("header"): "450", q("footer"): "450", q("gutter"): "0"})
    return ET.tostring(doc, encoding="utf-8", xml_declaration=True)


def main():
    markdown = SOURCE.read_text(encoding="utf-8")
    content_types = ET.Element(f"{{{CT}}}Types")
    ET.SubElement(content_types, f"{{{CT}}}Default", {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"})
    ET.SubElement(content_types, f"{{{CT}}}Default", {"Extension": "xml", "ContentType": "application/xml"})
    ET.SubElement(content_types, f"{{{CT}}}Override", {"PartName": "/word/document.xml", "ContentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"})
    ET.SubElement(content_types, f"{{{CT}}}Override", {"PartName": "/word/styles.xml", "ContentType": "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"})
    ET.SubElement(content_types, f"{{{CT}}}Override", {"PartName": "/docProps/core.xml", "ContentType": "application/vnd.openxmlformats-package.core-properties+xml"})
    package_rels = ET.Element(f"{{{REL}}}Relationships")
    ET.SubElement(package_rels, f"{{{REL}}}Relationship", {"Id": "rId1", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", "Target": "word/document.xml"})
    word_rels = ET.Element(f"{{{REL}}}Relationships")
    ET.SubElement(word_rels, f"{{{REL}}}Relationship", {"Id": "rId1", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles", "Target": "styles.xml"})
    core = ET.Element("{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}coreProperties")
    ET.SubElement(core, f"{{{DC}}}title").text = "AgapAI Spatiotemporal Function-by-Function Study Notes"
    ET.SubElement(core, f"{{{DC}}}subject").text = "Location retrieval, date/time retrieval, and event clustering"
    ET.SubElement(core, f"{{{DC}}}creator").text = "AgapAI Study Notes"
    ET.SubElement(core, f"{{{DCTERMS}}}created", {f"{{{XSI}}}type": "dcterms:W3CDTF"}).text = "2026-10-29T00:00:00Z"

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", ET.tostring(content_types, encoding="utf-8", xml_declaration=True))
        package.writestr("_rels/.rels", ET.tostring(package_rels, encoding="utf-8", xml_declaration=True))
        package.writestr("word/_rels/document.xml.rels", ET.tostring(word_rels, encoding="utf-8", xml_declaration=True))
        package.writestr("word/document.xml", document_xml(markdown))
        package.writestr("word/styles.xml", styles_xml())
        package.writestr("docProps/core.xml", ET.tostring(core, encoding="utf-8", xml_declaration=True))

    with zipfile.ZipFile(OUTPUT) as package:
        bad = package.testzip()
        if bad:
            raise RuntimeError(f"Corrupt DOCX component: {bad}")
        ET.fromstring(package.read("word/document.xml"))
        ET.fromstring(package.read("word/styles.xml"))
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
