import os
import pandas as pd
import re
from tqdm import tqdm

# ───────────────────────────────
# CONFIGURATION
# ───────────────────────────────
root = "/mnt/DATA1/vivekananda/DATA/iux_data"
#image_dir = "/mnt/DATA1/vivekananda/DATA/iux_data/images"
image_dir = os.path.join(root, "images")
# Input and output paths
input_tsv = os.path.join(root, "reports.tsv")
output_tsv = os.path.join(root, "reports_clean.tsv")


# ───────────────────────────────
# TEXT CLEANING FUNCTIONS
# ───────────────────────────────

def lowercase(text):
    new_text = []
    for line in text:
        new_text.append(str(line).lower())
    return new_text

def decontractions(text):
    new_text = []
    for phrase in text:
        phrase = re.sub(r"won't", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"couldn\'t", "could not", phrase)
        phrase = re.sub(r"shouldn\'t", "should not", phrase)
        phrase = re.sub(r"wouldn\'t", "would not", phrase)
        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        phrase = re.sub(r"\*+", "abuse", phrase)
        new_text.append(phrase)
    return new_text

def rem_punctuations(text):
    punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*~'''
    new_text = []
    for line in text:
        for char in line:
            if char in punctuations:
                line = line.replace(char, "")
        new_text.append(" ".join(e for e in line.split()))
    return new_text

def rem_numbers(text):
    new_text = []
    for line in text:
        temp = re.sub(r"x*", "", line)
        new_text.append(re.sub(r"\d", "", temp))
    return new_text

def words_filter(text):
    new_text = []
    for line in text:
        temp = line.split()
        temp2 = []
        for word in temp:
            if len(word) <= 2 and word not in ["no", "ct"]:
                continue
            else:
                temp2.append(word)
        new_text.append(" ".join(e for e in temp2))
    return new_text

def multiple_fullstops(text):
    new_text = []
    for line in text:
        new_text.append(re.sub(r"\.\.+", ".", line))
    return new_text

def fullstops(text):
    new_text = []
    for line in text:
        new_text.append(re.sub(r"\.", " .", line))
    return new_text

def multiple_spaces(text):
    new_text = []
    for line in text:
        new_text.append(" ".join(e for e in line.split()))
    return new_text

def separting_startg_words(text):
    new_text = []
    for line in text:
        temp = []
        words = line.split()
        for i in words:
            if not i.startswith("."):
                temp.append(i)
            else:
                w = i.replace(".", ". ")
                temp.append(w)
        new_text.append(" ".join(e for e in temp))
    return new_text

def rem_apostrophes(text):
    new_text = []
    for line in text:
        new_text.append(re.sub("'", "", line))
    return new_text

def text_preprocessing(text_series):
    """Combines all preprocessing steps on a Pandas Series."""
    text_list = list(text_series)
    text_list = lowercase(text_list)
    text_list = decontractions(text_list)
    text_list = rem_punctuations(text_list)
    text_list = rem_numbers(text_list)
    text_list = words_filter(text_list)
    text_list = multiple_fullstops(text_list)
    text_list = fullstops(text_list)
    text_list = multiple_spaces(text_list)
    text_list = separting_startg_words(text_list)
    text_list = rem_apostrophes(text_list)
    return text_list



# ───────────────────────────────
# LOAD EXISTING TSV FILE
# ───────────────────────────────
if not os.path.exists(input_tsv):
    raise FileNotFoundError(f"Input file not found: {input_tsv}")

df = pd.read_csv(input_tsv, sep="\t")
print(f"Loaded TSV: {input_tsv}")
print(f"Initial samples: {len(df)}")

# ───────────────────────────────
# HANDLE MISSING / INVALID REPORTS
# ───────────────────────────────
df["report"] = df["report"].astype(str)
df["report"].replace(["nan", "NaN", "None", "NULL", "null"], pd.NA, inplace=True)
df = df[df["report"].notna() & (df["report"].str.strip() != "")]

# ───────────────────────────────
# APPLY TEXT PREPROCESSING PIPELINE
# ───────────────────────────────
print("Running text preprocessing...")
df["report"] = text_preprocessing(df["report"])
print("Text preprocessing completed.")

# ───────────────────────────────
# VERIFY IMAGE EXISTENCE
# ───────────────────────────────
valid_rows = []
missing_images = 0

for _, row in tqdm(df.iterrows(), total=len(df), desc="Verifying images"):
    img_path = os.path.join(image_dir, row["image_name"])
    if os.path.exists(img_path):
        valid_rows.append(row)
    else:
        missing_images += 1

df = pd.DataFrame(valid_rows)

print(f"Final valid samples: {len(df)}")

# ───────────────────────────────
# SAVE CLEAN DATA
# ───────────────────────────────
# Save full cleaned dataset for analysis (with header)

# Save training TSV (no header)
df[["image_name", "report"]].to_csv(output_tsv, sep="\t", index=False, header=False, encoding="utf-8")
print(f"Saved training-ready TSV (no header): {output_tsv} ({len(df)} samples)")

# ───────────────────────────────
# SAMPLE OUTPUT
# ───────────────────────────────
print("\nSample cleaned entries (first 3):")
print(df.head(3))
