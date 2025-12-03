import os
import pandas as pd
import re
import chardet

ENRON_PATH = "data/kaggle_datasets/enron_email/emails.csv"
SMS_SPAM_PATH = "data/kaggle_datasets/sms_spam/spam.csv"
CHATGPT_DIR = "data/chatgpt_dataset"

OUT_CSV = "data/text_dataset.csv"


# -------------------------
# CLEAN TEXT
# -------------------------
def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------
# SAFE READ CSV
# -------------------------
def safe_read_csv(path):
    print(f"\n📄 Reading CSV safely: {path}")

    # Try UTF-8
    try:
        df = pd.read_csv(path, encoding="utf-8")
        print("✔ Loaded using UTF-8")
        return df
    except:
        print("❌ UTF-8 failed, detecting encoding...")

    # Detect encoding
    with open(path, "rb") as f:
        encoding = chardet.detect(f.read())["encoding"]

    print(f"🔍 Detected encoding: {encoding}")

    df = pd.read_csv(path, encoding=encoding)
    print(f"✔ Loaded using {encoding}")

    return df


# -------------------------
# MAIN BUILD LIST
# -------------------------
rows = []

# ------------------------------------
# LOAD ENRON EMAILS (BENIGN)
# ------------------------------------
if os.path.exists(ENRON_PATH):
    print("\n📥 Loading Enron Emails...")
    df_enron = safe_read_csv(ENRON_PATH)

    # If "text" exists, use it; else use last column
    text_col = "text" if "text" in df_enron.columns else df_enron.columns[-1]

    for text in df_enron[text_col].dropna():
        rows.append([clean(text), "benign"])


# ------------------------------------
# LOAD SMS SPAM COLLECTION
# ------------------------------------
if os.path.exists(SMS_SPAM_PATH):
    print("\n📥 Loading SMS Spam Dataset...")
    df_sms = safe_read_csv(SMS_SPAM_PATH)

    # Your file uses v1 = label, v2 = message
    text_col = "v2"
    label_col = "v1"

    for _, row in df_sms.iterrows():
        text = clean(str(row[text_col]))
        if not text:
            continue

        label_raw = str(row[label_col]).lower()
        if "spam" in label_raw:
            rows.append([text, "phishing"])
        else:
            rows.append([text, "benign"])


# ------------------------------------
# LOAD ChatGPT-generated CSV datasets
# ------------------------------------
print("\n📥 Loading ChatGPT Datasets...")

for fname in os.listdir(CHATGPT_DIR):
    if fname.endswith(".csv"):

        fpath = os.path.join(CHATGPT_DIR, fname)
        df = safe_read_csv(fpath)

        # Determine text column: search priority
        possible_cols = ["message", "text", "body", "content"]
        found_col = None

        for col in possible_cols:
            if col in df.columns:
                found_col = col
                break

        if found_col is None:
            # fallback: last column
            found_col = df.columns[-1]

        # Determine label type
        label = "phishing" if "phish" in fname.lower() else "benign"

        for text in df[found_col].dropna():
            text = clean(str(text))
            if text:
                rows.append([text, label])


# ------------------------------------
# SAVE FINAL MERGED DATASET
# ------------------------------------
df_out = pd.DataFrame(rows, columns=["text", "label"])
df_out.to_csv(OUT_CSV, index=False)

print("\n✅ Text dataset created successfully!")
print("📁 Saved to:", OUT_CSV)
print("📊 Total samples:", len(df_out))
print(df_out["label"].value_counts())
