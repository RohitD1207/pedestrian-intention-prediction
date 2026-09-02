import xml.etree.ElementTree as ET
import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
annotation_dir = PROJECT_ROOT / "data" / "PIE_clips" / "annotations" / "annotations" / "set01"

rows = []

for file in os.listdir(annotation_dir):

    if not file.endswith(".xml"):
        continue

    video_name = file.replace("_annt.xml","")

    tree = ET.parse(annotation_dir / file)
    root = tree.getroot()

    for track in root.findall("track"):

        if track.attrib["label"] != "pedestrian":
            continue

        for box in track.findall("box"):

            frame = int(box.attrib["frame"])

            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            attrs = {a.attrib["name"]: a.text for a in box.findall("attribute")}

            crossing = attrs.get("cross","not-crossing")

            label = 1 if crossing == "crossing" else 0

            rows.append({
                "video": video_name,
                "frame": frame,
                "pedestrian_id": attrs["id"],
                "x1": xtl,
                "y1": ytl,
                "x2": xbr,
                "y2": ybr,
                "label": label
            })

df = pd.DataFrame(rows)

df.to_csv(PROJECT_ROOT / "datasets" / "pie_annotations_set01.csv", index=False)

print("Saved cleaned annotations.")