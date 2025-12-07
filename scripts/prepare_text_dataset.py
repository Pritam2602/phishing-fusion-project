import os
import pandas as pd
import re
import chardet


SMS_SPAM_PATH = "data/kaggle_datasets/sms_spam/spam.csv"
CHATGPT_DIR = "data/chatgpt_dataset"
OUT_CSV = "data/text_dataset.csv"


def clean(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()



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


def extract_body(oneline_text):
    """
    Extracts Body=... from one-line email messages
    """
    if "body=" in oneline_text.lower():
        parts = re.split(r"body=", oneline_text, flags=re.I)
        return clean(parts[-1])
    return clean(oneline_text)


rows = []

print("\n==============================")
print("📥 Building Unified Dataset...")
print("==============================\n")





if os.path.exists(SMS_SPAM_PATH):
    print("\n📥 Loading SMS Spam Dataset...")
    df_sms = safe_read_csv(SMS_SPAM_PATH)

    text_col = "v2"
    label_col = "v1"

    for _, row in df_sms.iterrows():
        text = clean(str(row[text_col]))
        if not text:
            continue

        label_raw = str(row[label_col]).lower()
        label = "phishing" if "spam" in label_raw else "benign"
        rows.append([text, label])



print("\n📥 Loading ChatGPT Datasets...")

for fname in os.listdir(CHATGPT_DIR):
    if fname.endswith(".csv"):
        fpath = os.path.join(CHATGPT_DIR, fname)
        df = safe_read_csv(fpath)

        possible_cols = ["message", "text", "body", "content"]
        found_col = None

        for col in possible_cols:
            if col in df.columns:
                found_col = col
                break
        if found_col is None:
            found_col = df.columns[-1]

        label = "phishing" if "phish" in fname.lower() else "benign"

        for text in df[found_col].dropna():
            cleaned = clean(str(text))
            if cleaned:
                rows.append([cleaned, label])





df_out = pd.DataFrame(rows, columns=["text", "label"])
df_out.to_csv(OUT_CSV, index=False)

print("\n===================================")
print("✅ Unified Text Dataset Completed!")
print("📁 Saved to:", OUT_CSV)
print("📊 Total samples:", len(df_out))
print(df_out["label"].value_counts())
print("===================================\n")
