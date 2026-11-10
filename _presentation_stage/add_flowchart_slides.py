from pathlib import Path
import re
import zipfile
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
PPTX = ROOT / "AgapAI_Spatiotemporal_Detailed_Presentation.pptx"
TEMP = PPTX.with_suffix(".flow.tmp")
MEDIA = ROOT / "flowchart_media"
MEDIA.mkdir(exist_ok=True)

W, H = 1920, 1080
BG = "#F4F7FB"
NAVY = "#172E4D"
BODY = "#52677F"
BLUE = "#3A86D1"
BLUE_BG = "#DFEEFC"
PURPLE = "#8054C8"
PURPLE_BG = "#EFE8FC"
GREEN = "#398E60"
GREEN_BG = "#E1F3E8"
WHITE = "#FFFFFF"
GREY = "#AEBFD2"

FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 25)
FONT_SMALL = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 21)
FONT_BOLD = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 30)
FONT_TITLE = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 48)
FONT_SUBTITLE = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 25)
FONT_LABEL = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 28)

NS_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS_CT = "http://schemas.openxmlformats.org/package/2006/content-types"
ET.register_namespace("a", NS_A)
ET.register_namespace("p", NS_P)
ET.register_namespace("r", NS_R)


def text_width(draw, text, font):
    return draw.textbbox((0, 0), text, font=font)[2]


def wrap(draw, text, font, max_width):
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if line and text_width(draw, candidate, font) > max_width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def draw_card(draw, box, title, body, accent, fill, body_font=FONT_SMALL, radius=18):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=accent, width=3)
    draw.text((x1 + 24, y1 + 15), title, fill=NAVY, font=FONT_BOLD)
    lines = wrap(draw, body, body_font, x2 - x1 - 48)
    y = y1 + 56
    line_height = body_font.size + 5
    for line in lines:
        draw.text((x1 + 24, y), line, fill=BODY, font=body_font)
        y += line_height
    last_text_bottom = y - line_height + body_font.size if lines else y
    if last_text_bottom > y2 - 5:
        raise ValueError(f"Text overflow in card {title!r}: bottom {y} > {y2}")


def arrow(draw, start, end, color=NAVY, width=5, head=14):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if y2 >= y1:
        draw.polygon([(x2 - head, y2 - head), (x2 + head, y2 - head), (x2, y2)], fill=color)
    elif x2 >= x1:
        draw.polygon([(x2 - head, y2 - head), (x2 - head, y2 + head), (x2, y2)], fill=color)


def header(draw, title, subtitle, section):
    draw.rectangle((0, 0, W, 12), fill=section)
    draw.text((W // 2, 28), title, fill=NAVY, font=FONT_TITLE, anchor="mt")
    draw.text((W // 2, 91), subtitle, fill=BODY, font=FONT_SUBTITLE, anchor="mt")


def first_flow():
    image = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(image)
    header(d, "Spatiotemporal pipeline: post to saved results",
           "Location and date/time are processed for each selected post before clustering.", BLUE)

    draw_card(d, (185, 150, 1735, 250), "INPUT POSTS",
              "Collected post text, created_at (UTC), author identity, and disaster type. Process in batches of up to 100.",
              GREY, WHITE, FONT)

    # Branch from input into the two retrieval paths.
    d.line((960, 250, 960, 282), fill=NAVY, width=5)
    d.line((490, 282, 1430, 282), fill=NAVY, width=5)
    arrow(d, (490, 282), (490, 312), BLUE)
    arrow(d, (1430, 282), (1430, 312), PURPLE)
    d.rounded_rectangle((65, 315, 915, 365), radius=12, fill=BLUE_BG, outline=BLUE, width=3)
    d.text((490, 340), "LOCATION RETRIEVAL", fill=NAVY, font=FONT_LABEL, anchor="mm")
    d.rounded_rectangle((1005, 315, 1855, 365), radius=12, fill=PURPLE_BG, outline=PURPLE, width=3)
    d.text((1430, 340), "DATE / TIME RETRIEVAL", fill=NAVY, font=FONT_LABEL, anchor="mm")

    left = [
        ((65, 385, 915, 495), "1. Find candidate place mentions",
         "spaCy model ljvmiranda921/tl_calamancy_md (revision 0.2.0) proposes LOC/ORG entities and address-like spans."),
        ((65, 510, 915, 625), "2. Validate with the Cebu PSGC gazetteer",
         "Normalize names and use municipality context. A unique match resolves; multiple matches remain ambiguous."),
        ((65, 640, 915, 765), "3. Refine ambiguous post-text locations",
         "Use direct HIGH-confidence anchors posted within 6 hours. Average their coordinates; accept the nearest candidate only within 5 km."),
        ((65, 780, 915, 875), "4. If post text has no location, use author context",
         "Bio (MEDIUM); if no result, up to 10 earlier authored posts (LOW)."),
    ]
    for box, title, body in left:
        draw_card(d, box, title, body, BLUE, BLUE_BG)
    for y1, y2 in [(495, 510), (625, 640), (765, 780)]:
        arrow(d, (490, y1), (490, y2), BLUE, width=4, head=10)

    right = [
        ((1005, 385, 1855, 510), "1. Parse the original text",
         "Microsoft Recognizers-Text DateTime and dateparser use created_at as the relative-time reference in Asia/Manila. Skip ambiguous clock resolutions."),
        ((1005, 525, 1855, 660), "2. Translate only when needed",
         "If no explicit date/time is found, detect language with langdetect; translate locally with NLLB-200 to English, then parse again."),
        ((1005, 675, 1855, 875), "3. Fill missing date/time parts",
         "Full date and time found → use both.\nDate only or time only → fill the missing part from created_at.\nNo text result → use metadata date and time; if metadata is invalid, leave unresolved."),
    ]
    for box, title, body in right:
        draw_card(d, box, title, body, PURPLE, PURPLE_BG)
    for y1, y2 in [(510, 525), (660, 675)]:
        arrow(d, (1430, y1), (1430, y2), PURPLE, width=4, head=10)

    # Both branches converge on the saved per-post result.
    d.line((490, 875, 490, 910), fill=NAVY, width=5)
    d.line((1430, 875, 1430, 910), fill=NAVY, width=5)
    d.line((490, 910, 1430, 910), fill=NAVY, width=5)
    arrow(d, (960, 910), (960, 927), NAVY, width=5, head=12)
    draw_card(d, (185, 935, 1735, 1035), "SAVE POST RESULTS TO MONGODB",
              "Store locations and ambiguity, normalized event datetime, Manila-local date/time, source, and text/time snapshots for reprocessing.",
              GREY, WHITE, FONT)
    d.text((960, 1060), "Anchor refinement: 6-hour window  |  nearest candidate: within 5 km",
           fill=BODY, font=FONT_SMALL, anchor="mm")
    return image


def second_flow():
    image = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(image)
    header(d, "Event clustering: eligible posts to event summaries",
           "Clustering uses posted timestamps; extracted event date/time is used for the event summary.", GREEN)

    boxes = [
        ((180, 155, 1740, 330), "1. Select posts eligible for clustering",
         "Accepted disaster type + at least one resolved location with coordinates + valid created_at.\nChoose one location per post by confidence, then breadcrumb specificity."),
        ((180, 370, 1740, 535), "2. Compare each pair of eligible posts",
         "Posts match only when disaster types are the same, locations are within 2 km, AND created_at timestamps are within 6 hours.\nDistance uses the Haversine formula (Earth radius 6,371 km)."),
        ((180, 575, 1740, 755), "3. Group compatible posts with DBSCAN",
         "Build a binary matrix: compatible pair = 0; incompatible pair = 1.\nRun DBSCAN with precomputed distances, eps = 0.5, min_samples = 1. Links are transitive; a post can form a group by itself."),
        ((180, 795, 1740, 965), "4. Build event summaries and return results",
         "Place: highest-confidence member location. Time: earliest explicit/vague event datetime; otherwise earliest created_at.\nReturn event groups and post rows to the dashboard."),
    ]
    for box, title, body in boxes:
        draw_card(d, box, title, body, GREEN, GREEN_BG if box != boxes[-1][0] else WHITE, FONT)
    for y1, y2 in [(330, 370), (535, 575), (755, 795)]:
        arrow(d, (960, y1), (960, y2), GREEN, width=5, head=13)

    d.rounded_rectangle((75, 995, 1845, 1050), radius=15, fill="#FCE8E6", outline="#D8756B", width=2)
    d.text((960, 1022), "Thresholds: 5 km = ambiguous-location refinement   |   2 km = event-pair distance   |   6 hours = anchor and event-pair windows",
           fill=NAVY, font=FONT_SMALL, anchor="mm")
    return image


def add_picture(slide_root, rid, image_id, name):
    pic = ET.Element(f"{{{NS_P}}}pic")
    nv = ET.SubElement(pic, f"{{{NS_P}}}nvPicPr")
    ET.SubElement(nv, f"{{{NS_P}}}cNvPr", {"id": str(image_id), "name": name})
    cpic = ET.SubElement(nv, f"{{{NS_P}}}cNvPicPr")
    ET.SubElement(cpic, f"{{{NS_A}}}picLocks", {"noChangeAspect": "1"})
    ET.SubElement(nv, f"{{{NS_P}}}nvPr")
    blip_fill = ET.SubElement(pic, f"{{{NS_P}}}blipFill")
    ET.SubElement(blip_fill, f"{{{NS_A}}}blip", {f"{{{NS_R}}}embed": rid})
    stretch = ET.SubElement(blip_fill, f"{{{NS_A}}}stretch")
    ET.SubElement(stretch, f"{{{NS_A}}}fillRect")
    sppr = ET.SubElement(pic, f"{{{NS_P}}}spPr")
    xfrm = ET.SubElement(sppr, f"{{{NS_A}}}xfrm")
    ET.SubElement(xfrm, f"{{{NS_A}}}off", {"x": "0", "y": "0"})
    ET.SubElement(xfrm, f"{{{NS_A}}}ext", {"cx": "12192000", "cy": "6858000"})
    geom = ET.SubElement(sppr, f"{{{NS_A}}}prstGeom", {"prst": "rect"})
    ET.SubElement(geom, f"{{{NS_A}}}avLst")
    root = slide_root.find(f".//{{{NS_P}}}spTree")
    ext = root.find(f"{{{NS_P}}}extLst")
    root.insert(list(root).index(ext), pic) if ext is not None else root.append(pic)


def add_relationship(package, slide_num, filename):
    path = f"ppt/slides/_rels/slide{slide_num}.xml.rels"
    rel_root = ET.fromstring(package[path])
    target = f"../media/{filename}"
    existing = next((rel for rel in rel_root if rel.attrib.get("Target") == target), None)
    if existing is not None:
        return path, existing.attrib["Id"], ET.tostring(rel_root, encoding="utf-8", xml_declaration=True)
    ids = [int(m.group(1)) for rel in rel_root for m in [re.fullmatch(r"rId(\d+)", rel.attrib.get("Id", ""))] if m]
    rid = f"rId{max(ids, default=0) + 1}"
    ET.SubElement(rel_root, f"{{{NS_REL}}}Relationship", {
        "Id": rid,
        "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
        "Target": target,
    })
    return path, rid, ET.tostring(rel_root, encoding="utf-8", xml_declaration=True)


def main():
    flow1 = first_flow()
    flow2 = second_flow()
    flow1_path = MEDIA / "spatiotemporal_retrieval.png"
    flow2_path = MEDIA / "event_clustering.png"
    flow1.save(flow1_path, optimize=True)
    flow2.save(flow2_path, optimize=True)

    replacements = {
        "ppt/media/spatiotemporal_retrieval.png": flow1_path.read_bytes(),
        "ppt/media/event_clustering.png": flow2_path.read_bytes(),
    }
    with zipfile.ZipFile(PPTX, "r") as source:
        package = {item.filename: source.read(item.filename) for item in source.infolist()}

    for slide_num, media_path, image_id, title in [
        (2, "ppt/media/spatiotemporal_retrieval.png", 501, "Spatiotemporal retrieval flowchart"),
        (3, "ppt/media/event_clustering.png", 502, "Event clustering flowchart"),
    ]:
        filename = Path(media_path).name
        rel_path, rid, rel_data = add_relationship(package, slide_num, filename)
        package[rel_path] = rel_data
        slide_path = f"ppt/slides/slide{slide_num}.xml"
        slide_root = ET.fromstring(package[slide_path])
        sp_tree = slide_root.find(f".//{{{NS_P}}}spTree")
        for old_pic in list(sp_tree.findall(f"{{{NS_P}}}pic")):
            nv_name = old_pic.find(f".//{{{NS_P}}}cNvPr")
            if nv_name is not None and nv_name.attrib.get("name") == title:
                sp_tree.remove(old_pic)
        add_picture(slide_root, rid, image_id, title)
        package[slide_path] = ET.tostring(slide_root, encoding="utf-8", xml_declaration=True)
        package[media_path] = replacements[media_path]

    types = ET.fromstring(package["[Content_Types].xml"])
    if not any(el.attrib.get("Extension", "").lower() == "png" for el in types):
        ET.SubElement(types, f"{{{NS_CT}}}Default", {"Extension": "png", "ContentType": "image/png"})
    package["[Content_Types].xml"] = ET.tostring(types, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(TEMP, "w", zipfile.ZIP_DEFLATED) as target:
        for name, data in package.items():
            target.writestr(name, data)
    TEMP.replace(PPTX)

    with zipfile.ZipFile(PPTX) as final:
        bad = final.testzip()
        if bad:
            raise RuntimeError(f"Corrupt PPTX component: {bad}")
        for slide_num in (2, 3):
            root = ET.fromstring(final.read(f"ppt/slides/slide{slide_num}.xml"))
            pics = root.findall(f".//{{{NS_P}}}pic")
            if not pics:
                raise RuntimeError(f"Flowchart image missing from slide {slide_num}")
    print(f"Added flowcharts to slides 2 and 3: {PPTX}")


if __name__ == "__main__":
    main()
