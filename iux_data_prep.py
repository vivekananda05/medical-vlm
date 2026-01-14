import os
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ───────────────────────────────
# CONFIGURATION
# ───────────────────────────────
root = "iux_data"  # output directory
image_dir = r"NLMCXR_png"  # path to your images
report_dir = r"NLMCXR_reports/ecgen-radiology"  # path to your XML reports

output_csv = os.path.join(root, "reports_detailed.csv")
output_tsv = os.path.join(root, "reports_brief.tsv")

os.makedirs(root, exist_ok=True)

columns = [
    "image_name",
    "caption",
    "comparison",
    "indication",
    "findings",
    "impression",
    "height",
    "width"
]

all_rows = []  # collect all data rows

xml_files = [f for f in os.listdir(report_dir) if f.endswith(".xml")]
print(f"Found {len(xml_files)} XML reports. Parsing detailed IU X-Ray XML...")

# ───────────────────────────────
# PARSE EACH XML REPORT
# ───────────────────────────────
for file in tqdm(xml_files, desc="Parsing IU X-Ray XML"):
    path = os.path.join(report_dir, file)

    try:
        mytree = ET.parse(path)
        root_xml = mytree.getroot()

        # Helper to safely extract XML text
        def extract_text(label):
            elem = root_xml.find(f".//AbstractText[@Label='{label}']")
            return elem.text.strip() if elem is not None and elem.text else ""

        comparison = extract_text("COMPARISON")
        indication = extract_text("INDICATION")
        findings = extract_text("FINDINGS")
        impression = extract_text("IMPRESSION")

        # Each XML can contain multiple images
        for x in root_xml.findall("parentImage"):
            image_id = x.attrib.get("id", None)
            if not image_id:
                continue

            img_name = image_id + ".png"
            img_path = os.path.join(image_dir, img_name)

            # fallback to .jpg if not found
            if not os.path.exists(img_path):
                img_name = image_id + ".jpg"
                img_path = os.path.join(image_dir, img_name)

            if not os.path.exists(img_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            height, width, _ = image.shape

            caption_elem = x.find("caption")
            caption = caption_elem.text.strip() if caption_elem is not None and caption_elem.text else ""

            all_rows.append({
                "image_name": img_name,
                "caption": caption,
                "comparison": comparison,
                "indication": indication,
                "findings": findings,
                "impression": impression,
                "height": height,
                "width": width
            })

    except Exception as e:
        print(f" Failed to parse {file}: {e}")

# ───────────────────────────────
# CREATE DATAFRAMES AND SAVE FILES
# ───────────────────────────────
df = pd.DataFrame(all_rows, columns=columns)
df.drop_duplicates(subset=["image_name"], inplace=True)
print(len(df))

# Save full CSV
df.to_csv(output_csv, index=False, encoding="utf-8")
print(f" Created detailed CSV: {output_csv} with {len(df)} entries")

# Save simplified TSV with only image_name and findings as report
df_brief = df[["image_name", "findings"]].rename(columns={"findings": "report"})
df_brief.to_csv(output_tsv, sep="\t", index=False, encoding="utf-8")
print(f" Created brief TSV: {output_tsv} with {len(df_brief)} entries")

# Show sample output
print("\nSample entries from brief TSV:")
print(df_brief.head(5))
